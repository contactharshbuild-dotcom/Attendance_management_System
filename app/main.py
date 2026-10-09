"""
Employee Attendance & Analytics API

Run:  uvicorn app.main:app --port 8000
Env:  MONGO_URI, MONGO_DB (read from environment, local .env loaded as fallback)
"""
import calendar
import json
import os
import re
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from bson import json_util
from bson.son import SON
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Path, Query, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator, model_validator
from pymongo import MongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError, PyMongoError

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "attendance_db")

client: MongoClient = MongoClient(MONGO_URI, tz_aware=True)
db = client[MONGO_DB]

IST = timezone(timedelta(hours=5, minutes=30))

MIN_EPOCH_MS = 100_000_000_000
MAX_EPOCH_MS = 4_102_444_800_000


# --------------------------------------------------------------------------- #
# Database Initialization & Lifespan
# --------------------------------------------------------------------------- #
def init_indexes():
    """Create all required indexes idempotently at startup."""
    # employees indexes
    db.employees.create_index([("emp_code", 1)], unique=True)
    db.employees.create_index([("department", 1), ("joined_on", 1)])
    db.employees.create_index([("joined_on", 1)])

    # attendance_logs indexes
    db.attendance_logs.create_index([("emp_code", 1), ("date", 1)], unique=True)
    db.attendance_logs.create_index([("date", -1), ("emp_code", 1)])
    db.attendance_logs.create_index([("emp_code", 1), ("date", -1)])
    db.attendance_logs.create_index([("emp_code", 1), ("punch_out", 1), ("punch_in", -1)])
    db.attendance_logs.create_index([("date", 1), ("late_minutes", -1)])
    db.attendance_logs.create_index([("date", 1), ("emp_code", 1)])


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_indexes()
    yield


app = FastAPI(
    title="Employee Attendance & Analytics API",
    version="2.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Precision Rounding & Date/Time Helpers
# --------------------------------------------------------------------------- #
def round_half_up(value: float, decimals: int = 2) -> float:
    """R8 Rounding: exact half-up rounding using Decimal."""
    d = Decimal(str(value))
    q = Decimal("10") ** (-decimals)
    return float(d.quantize(q, rounding=ROUND_HALF_UP))


def to_epoch_ms(dt: Optional[datetime]) -> Optional[int]:
    """Convert a UTC BSON datetime to integer epoch milliseconds."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def to_bson_datetime(ms: Optional[int]) -> Optional[datetime]:
    """Convert epoch milliseconds to a whole-second truncated UTC datetime."""
    if ms is None:
        return None
    # R1: Instants are truncated to whole seconds before calculation and storage
    seconds = ms // 1000
    return datetime.fromtimestamp(seconds, tz=timezone.utc)


def ist_from_ms(ms: int) -> datetime:
    """Return timezone-aware datetime in IST from epoch milliseconds."""
    seconds = ms // 1000
    return datetime.fromtimestamp(seconds, tz=timezone.utc).astimezone(IST)


def compute_attendance_date(punch_in_ms: int, shift_start: str, shift_end: str) -> str:
    """
    R1: Attendance date is the IST calendar date of punch_in.
    For overnight shifts (shift_end <= shift_start), a punch-in whose IST time
    is strictly earlier than shift_end belongs to the shift that started the previous day.
    """
    dt_ist = ist_from_ms(punch_in_ms)
    if shift_end <= shift_start:
        end_h, end_m = map(int, shift_end.split(":"))
        punch_seconds = dt_ist.hour * 3600 + dt_ist.minute * 60 + dt_ist.second
        shift_end_seconds = end_h * 3600 + end_m * 60
        if punch_seconds < shift_end_seconds:
            return (dt_ist.date() - timedelta(days=1)).isoformat()
    return dt_ist.date().isoformat()


def compute_late_minutes(punch_in_ms: int, shift_start: str, shift_end: str, attendance_date: str) -> int:
    """
    R2: 10-minute grace period. Late only if punch_in is strictly more than 10:00 (600s)
    after shift_start; then late_minutes = whole minutes (floor) since shift_start.
    """
    start_h, start_m = map(int, shift_start.split(":"))
    shift_start_dt = datetime.fromisoformat(attendance_date).replace(
        hour=start_h, minute=start_m, second=0, microsecond=0, tzinfo=IST
    )
    punch_in_dt = ist_from_ms(punch_in_ms)
    elapsed_seconds = (punch_in_dt - shift_start_dt).total_seconds()
    if elapsed_seconds > 600:
        return int(elapsed_seconds // 60)
    return 0


def compute_work_hours(punch_in_ms: int, punch_out_ms: int) -> float:
    """
    R4: (punch_out - punch_in) in seconds / 3600, rounded to 2 decimals half-up.
    Instants are truncated to whole seconds.
    """
    in_sec = punch_in_ms // 1000
    out_sec = punch_out_ms // 1000
    diff_sec = out_sec - in_sec
    val = (Decimal(diff_sec) / Decimal(3600)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return float(val)


def compute_overtime(punch_out_ms: int, shift_start: str, shift_end: str, attendance_date: str) -> int:
    """
    R3: Whole minutes (floor) between shift_end and punch_out, counted only if >= 30; otherwise 0.
    For overnight shifts, shift_end falls on the next calendar day.
    """
    start_date = datetime.fromisoformat(attendance_date).date()
    if shift_end <= shift_start:
        end_date = start_date + timedelta(days=1)
    else:
        end_date = start_date

    end_h, end_m = map(int, shift_end.split(":"))
    shift_end_dt = datetime(end_date.year, end_date.month, end_date.day, end_h, end_m, 0, tzinfo=IST)
    punch_out_dt = ist_from_ms(punch_out_ms)
    diff_seconds = (punch_out_dt - shift_end_dt).total_seconds()
    diff_minutes = int(diff_seconds // 60)
    return diff_minutes if diff_minutes >= 30 else 0


def count_working_days(start_date_str: str, end_date_str: str) -> int:
    """Count Monday-Friday days between start_date and end_date inclusive."""
    start_d = date.fromisoformat(start_date_str)
    end_d = date.fromisoformat(end_date_str)
    if start_d > end_d:
        return 0
    cur = start_d
    count = 0
    while cur <= end_d:
        if cur.weekday() < 5:  # Mon=0, Fri=4
            count += 1
        cur += timedelta(days=1)
    return count


def format_attendance_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Convert MongoDB attendance log document to API AttendanceRecord format."""
    p_in = to_epoch_ms(doc.get("punch_in"))
    p_out = to_epoch_ms(doc.get("punch_out"))

    formatted_history = []
    for h in doc.get("history") or []:
        h_at = to_epoch_ms(h.get("at")) if isinstance(h.get("at"), datetime) else h.get("at")
        changes = {}
        for k, v in (h.get("changes") or {}).items():
            f_val = v.get("from")
            t_val = v.get("to")
            if k in ("punch_in", "punch_out"):
                if isinstance(f_val, datetime):
                    f_val = to_epoch_ms(f_val)
                if isinstance(t_val, datetime):
                    t_val = to_epoch_ms(t_val)
            changes[k] = {"from": f_val, "to": t_val}
        formatted_history.append({
            "at": h_at,
            "by": h.get("by"),
            "reason": h.get("reason"),
            "changes": changes,
        })

    return {
        "emp_code": doc["emp_code"],
        "date": doc["date"],
        "status": doc["status"],
        "punch_in": p_in,
        "punch_out": p_out,
        "work_hours": doc.get("work_hours"),
        "late_minutes": doc.get("late_minutes", 0),
        "overtime_minutes": doc.get("overtime_minutes", 0),
        "half_day": doc.get("half_day", False),
        "history": formatted_history,
    }


def format_employee_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Convert MongoDB employee document to API Employee format."""
    created_at_val = doc.get("created_at")
    if isinstance(created_at_val, datetime):
        created_at_ms = to_epoch_ms(created_at_val)
    elif isinstance(created_at_val, (int, float)):
        created_at_ms = int(created_at_val)
    else:
        created_at_ms = None

    return {
        "emp_code": doc["emp_code"],
        "name": doc["name"],
        "email": doc["email"],
        "department": doc["department"],
        "shift_start": doc.get("shift_start", "09:30"),
        "shift_end": doc.get("shift_end", "18:30"),
        "joined_on": doc["joined_on"],
        "created_at": created_at_ms,
    }


# --------------------------------------------------------------------------- #
# Pydantic Schemas
# --------------------------------------------------------------------------- #
class EmployeeCreate(BaseModel):
    emp_code: str = Field(..., pattern=r"^EMP\d{4,6}$")
    name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    department: str = Field(..., min_length=1, max_length=50)
    shift_start: str = Field("09:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    shift_end: str = Field("18:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    joined_on: str = Field(...)

    @field_validator("joined_on")
    @classmethod
    def validate_joined_on(cls, v: str):
        try:
            date.fromisoformat(v)
        except ValueError:
            raise ValueError("joined_on must be YYYY-MM-DD")
        return v

    @model_validator(mode="after")
    def validate_shifts(self):
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start must differ from shift_end")
        return self


class PunchInRequest(BaseModel):
    emp_code: str
    punched_at: Optional[int] = Field(None, ge=MIN_EPOCH_MS, le=MAX_EPOCH_MS)
    status: str = Field("PRESENT", pattern=r"^(PRESENT|WFH|ON_DUTY)$")


class PunchOutRequest(BaseModel):
    emp_code: str
    punched_at: Optional[int] = Field(None, ge=MIN_EPOCH_MS, le=MAX_EPOCH_MS)


class RegularizeRequest(BaseModel):
    status: Optional[str] = Field(None, pattern=r"^(PRESENT|ABSENT|LEAVE|WFH|ON_DUTY)$")
    punch_in: Optional[int] = Field(None, ge=MIN_EPOCH_MS, le=MAX_EPOCH_MS)
    punch_out: Optional[int] = Field(None, ge=MIN_EPOCH_MS, le=MAX_EPOCH_MS)
    reason: str = Field(..., min_length=5, max_length=200)
    regularized_by: str = Field(..., min_length=1, max_length=50)


# --------------------------------------------------------------------------- #
# System Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    try:
        client.admin.command("ping")
        return {"status": "ok"}
    except Exception:
        raise HTTPException(status_code=503, detail="Database unreachable")


# --------------------------------------------------------------------------- #
# Employee Endpoints
# --------------------------------------------------------------------------- #
@app.post("/employees", status_code=status.HTTP_201_CREATED)
def create_employee(body: EmployeeCreate):
    created_at_dt = datetime.now(timezone.utc).replace(microsecond=0)
    doc = body.model_dump()
    doc["created_at"] = created_at_dt

    try:
        db.employees.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="emp_code already exists")

    return format_employee_doc(doc)


@app.get("/employees")
def list_employees(
    department: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    q = {}
    if department:
        q["department"] = department

    total = db.employees.count_documents(q)
    skip = (page - 1) * page_size
    cursor = db.employees.find(q).sort("emp_code", 1).skip(skip).limit(page_size)
    items = [format_employee_doc(d) for d in cursor]
    return {"items": items, "total": total, "page": page, "page_size": page_size}


# --------------------------------------------------------------------------- #
# Attendance Endpoints
# --------------------------------------------------------------------------- #
@app.post("/attendance/punch-in", status_code=status.HTTP_201_CREATED)
def punch_in(body: PunchInRequest):
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    punch_ms = body.punched_at if body.punched_at is not None else int(datetime.now(timezone.utc).timestamp() * 1000)
    # R1: Instants are truncated to whole seconds
    punch_ms = (punch_ms // 1000) * 1000
    punch_dt = to_bson_datetime(punch_ms)

    att_date = compute_attendance_date(punch_ms, emp["shift_start"], emp["shift_end"])
    late_min = compute_late_minutes(punch_ms, emp["shift_start"], emp["shift_end"], att_date)

    doc = {
        "emp_code": body.emp_code,
        "date": att_date,
        "status": body.status,
        "punch_in": punch_dt,
        "punch_out": None,
        "work_hours": None,
        "late_minutes": late_min,
        "overtime_minutes": 0,
        "half_day": False,
        "history": [],
    }

    try:
        db.attendance_logs.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="Already punched in for this date")

    return format_attendance_doc(doc)


@app.post("/attendance/punch-out")
def punch_out(body: PunchOutRequest):
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    punch_out_ms = body.punched_at if body.punched_at is not None else int(datetime.now(timezone.utc).timestamp() * 1000)
    punch_out_ms = (punch_out_ms // 1000) * 1000
    punch_out_dt = to_bson_datetime(punch_out_ms)

    # Find the most recent record whose punch_in <= punched_at
    rec = db.attendance_logs.find_one(
        {"emp_code": body.emp_code, "punch_in": {"$lte": punch_out_dt}},
        sort=[("punch_in", -1)],
    )
    if not rec:
        raise HTTPException(status_code=404, detail="No punch-in found")

    if rec.get("punch_out") is not None:
        raise HTTPException(status_code=409, detail="That record is already punched out")

    punch_in_dt = rec["punch_in"]
    punch_in_ms = to_epoch_ms(punch_in_dt)

    # 422 if punched_at <= punch_in or more than 24 hours after it
    if punch_out_ms <= punch_in_ms:
        raise HTTPException(status_code=422, detail="punch_out must be strictly after punch_in")
    if (punch_out_ms - punch_in_ms) > 24 * 3600 * 1000:
        raise HTTPException(status_code=422, detail="punch_out cannot be more than 24 hours after punch_in")

    work_hrs = compute_work_hours(punch_in_ms, punch_out_ms)
    overtime_min = compute_overtime(punch_out_ms, emp["shift_start"], emp["shift_end"], rec["date"])
    half_day_val = bool(work_hrs < 4.50)

    # Atomic update to avoid race conditions (concurrent punch-outs)
    updated = db.attendance_logs.find_one_and_update(
        {"_id": rec["_id"], "punch_out": None},
        {
            "$set": {
                "punch_out": punch_out_dt,
                "work_hours": work_hrs,
                "overtime_minutes": overtime_min,
                "half_day": half_day_val,
            }
        },
        return_document=ReturnDocument.AFTER,
    )
    if not updated:
        raise HTTPException(status_code=409, detail="That record is already punched out")

    return format_attendance_doc(updated)


@app.get("/attendance")
def list_attendance(
    emp_code: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[str] = Query(None, pattern=r"^(PRESENT|ABSENT|LEAVE|WFH|ON_DUTY)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    if date_from and date_to and date_from > date_to:
        raise HTTPException(status_code=422, detail="date_from must be <= date_to")

    q: Dict[str, Any] = {}
    if emp_code:
        q["emp_code"] = emp_code
    if date_from or date_to:
        q["date"] = {}
        if date_from:
            q["date"]["$gte"] = date_from
        if date_to:
            q["date"]["$lte"] = date_to
    if status:
        q["status"] = status

    total = db.attendance_logs.count_documents(q)
    skip = (page - 1) * page_size
    cursor = db.attendance_logs.find(q).sort([("date", -1), ("emp_code", 1)]).skip(skip).limit(page_size)
    items = [format_attendance_doc(d) for d in cursor]
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@app.patch("/attendance/{emp_code}/{date}")
def regularize_attendance(
    emp_code: str = Path(...),
    date_val: str = Path(..., alias="date"),
    body: RegularizeRequest = ...,
):
    emp = db.employees.find_one({"emp_code": emp_code})
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    rec = db.attendance_logs.find_one({"emp_code": emp_code, "date": date_val})
    if not rec:
        raise HTTPException(status_code=404, detail="Attendance record not found for this date")

    # Target values
    target_status = body.status if body.status is not None else rec["status"]

    target_punch_in_ms = (
        body.punch_in
        if body.punch_in is not None
        else to_epoch_ms(rec.get("punch_in"))
    )
    if target_punch_in_ms is not None:
        target_punch_in_ms = (target_punch_in_ms // 1000) * 1000

    target_punch_out_ms = (
        body.punch_out
        if body.punch_out is not None
        else to_epoch_ms(rec.get("punch_out"))
    )
    if target_punch_out_ms is not None:
        target_punch_out_ms = (target_punch_out_ms // 1000) * 1000

    # Rules validation
    if target_status in ("ABSENT", "LEAVE"):
        # Setting ABSENT or LEAVE clears punch times; supplying punch_in or punch_out with it is 422
        if body.punch_in is not None or body.punch_out is not None:
            raise HTTPException(status_code=422, detail="punch_in and punch_out cannot be supplied for ABSENT or LEAVE")
        target_punch_in_ms = None
        target_punch_out_ms = None
    else:
        # Presence status requires punch_in
        if target_punch_in_ms is None:
            raise HTTPException(status_code=422, detail="A presence status requires a punch_in")

        # punch_in must stay on the record's attendance date per R1
        computed_date = compute_attendance_date(target_punch_in_ms, emp["shift_start"], emp["shift_end"])
        if computed_date != date_val:
            raise HTTPException(status_code=422, detail="punch_in must belong to this record's attendance date")

        # punch_out checks
        if target_punch_out_ms is not None:
            if target_punch_out_ms <= target_punch_in_ms:
                raise HTTPException(status_code=422, detail="punch_out must be strictly after punch_in")
            if (target_punch_out_ms - target_punch_in_ms) > 24 * 3600 * 1000:
                raise HTTPException(status_code=422, detail="punch_out cannot be more than 24 hours after punch_in")

    # Compute derived fields
    if target_status in ("ABSENT", "LEAVE"):
        target_work_hours = None
        target_late_minutes = 0
        target_overtime_minutes = 0
        target_half_day = False
    else:
        target_late_minutes = compute_late_minutes(target_punch_in_ms, emp["shift_start"], emp["shift_end"], date_val)
        if target_punch_out_ms is not None:
            target_work_hours = compute_work_hours(target_punch_in_ms, target_punch_out_ms)
            target_overtime_minutes = compute_overtime(target_punch_out_ms, emp["shift_start"], emp["shift_end"], date_val)
            target_half_day = bool(target_work_hours < 4.50)
        else:
            target_work_hours = None
            target_overtime_minutes = 0
            target_half_day = False

    # Old values
    old_punch_in_ms = to_epoch_ms(rec.get("punch_in"))
    old_punch_out_ms = to_epoch_ms(rec.get("punch_out"))
    old_status = rec["status"]
    old_work_hours = rec.get("work_hours")
    old_late_minutes = rec.get("late_minutes", 0)
    old_overtime_minutes = rec.get("overtime_minutes", 0)
    old_half_day = rec.get("half_day", False)

    # Detect actual changes
    db_changes: Dict[str, Any] = {}
    if target_status != old_status:
        db_changes["status"] = {"from": old_status, "to": target_status}
    if target_punch_in_ms != old_punch_in_ms:
        db_changes["punch_in"] = {
            "from": rec.get("punch_in"),
            "to": to_bson_datetime(target_punch_in_ms),
        }
    if target_punch_out_ms != old_punch_out_ms:
        db_changes["punch_out"] = {
            "from": rec.get("punch_out"),
            "to": to_bson_datetime(target_punch_out_ms),
        }
    if target_work_hours != old_work_hours:
        db_changes["work_hours"] = {"from": old_work_hours, "to": target_work_hours}
    if target_late_minutes != old_late_minutes:
        db_changes["late_minutes"] = {"from": old_late_minutes, "to": target_late_minutes}
    if target_overtime_minutes != old_overtime_minutes:
        db_changes["overtime_minutes"] = {"from": old_overtime_minutes, "to": target_overtime_minutes}
    if target_half_day != old_half_day:
        db_changes["half_day"] = {"from": old_half_day, "to": target_half_day}

    if not db_changes:
        raise HTTPException(status_code=422, detail="No fields were changed by this request")

    now_dt = datetime.now(timezone.utc).replace(microsecond=0)
    history_entry = {
        "at": now_dt,
        "by": body.regularized_by,
        "reason": body.reason,
        "changes": db_changes,
    }

    # Concurrency control: check history length to avoid losing concurrent history entries
    cur_history_len = len(rec.get("history") or [])
    updated = db.attendance_logs.find_one_and_update(
        {
            "_id": rec["_id"],
            f"history.{cur_history_len}": {"$exists": False},
        },
        {
            "$set": {
                "status": target_status,
                "punch_in": to_bson_datetime(target_punch_in_ms),
                "punch_out": to_bson_datetime(target_punch_out_ms),
                "work_hours": target_work_hours,
                "late_minutes": target_late_minutes,
                "overtime_minutes": target_overtime_minutes,
                "half_day": target_half_day,
            },
            "$push": {
                "history": history_entry,
            },
        },
        return_document=ReturnDocument.AFTER,
    )
    if not updated:
        raise HTTPException(status_code=409, detail="Concurrent modification conflict")

    return format_attendance_doc(updated)


# --------------------------------------------------------------------------- #
# Analytics Endpoints
# --------------------------------------------------------------------------- #
@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(
    emp_code: str = Path(...),
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
):
    emp = db.employees.find_one({"emp_code": emp_code})
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    year, m = map(int, month.split("-"))
    num_days = calendar.monthrange(year, m)[1]
    month_start = f"{month}-01"
    month_end = f"{month}-{num_days:02d}"

    working_days_start = max(month_start, emp["joined_on"])
    working_days = count_working_days(working_days_start, month_end)

    pipeline = [
        {
            "$match": {
                "emp_code": emp_code,
                "date": {"$gte": month_start, "$lte": month_end},
            }
        },
        {
            "$project": {
                "status": 1,
                "date": 1,
                "half_day": {"$ifNull": ["$half_day", False]},
                "late_minutes": {"$ifNull": ["$late_minutes", 0]},
                "overtime_minutes": {"$ifNull": ["$overtime_minutes", 0]},
                "is_weekday": {
                    "$in": [
                        {"$dayOfWeek": {"$dateFromString": {"dateString": "$date"}}},
                        [2, 3, 4, 5, 6],
                    ]
                },
                "is_present": {"$in": ["$status", ["PRESENT", "WFH", "ON_DUTY"]]},
            }
        },
        {
            "$group": {
                "_id": "$emp_code",
                "present_days": {
                    "$sum": {
                        "$cond": [
                            {"$and": ["$is_weekday", "$is_present"]},
                            {"$cond": ["$half_day", 0.5, 1.0]},
                            0.0,
                        ]
                    }
                },
                "leave_days": {
                    "$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}
                },
                "late_count": {
                    "$sum": {"$cond": [{"$gt": ["$late_minutes", 0]}, 1, 0]}
                },
                "total_late_minutes": {"$sum": "$late_minutes"},
                "total_overtime_minutes": {"$sum": "$overtime_minutes"},
            }
        },
    ]

    res = list(db.attendance_logs.aggregate(pipeline))
    if res:
        data = res[0]
        present_days = round_half_up(data["present_days"], 2)
        leave_days = int(data["leave_days"])
        late_count = int(data["late_count"])
        total_late_minutes = int(data["total_late_minutes"])
        total_overtime_minutes = int(data["total_overtime_minutes"])
    else:
        present_days = 0.0
        leave_days = 0
        late_count = 0
        total_late_minutes = 0
        total_overtime_minutes = 0

    if working_days > 0:
        attendance_pct = round_half_up((present_days / working_days) * 100, 2)
    else:
        attendance_pct = None

    return {
        "emp_code": emp_code,
        "month": month,
        "working_days": working_days,
        "present_days": present_days,
        "leave_days": leave_days,
        "late_count": late_count,
        "total_late_minutes": total_late_minutes,
        "total_overtime_minutes": total_overtime_minutes,
        "attendance_pct": attendance_pct,
    }


@app.get("/analytics/departments/summary")
def department_summary(
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    department: Optional[str] = Query(None),
):
    year, m = map(int, month.split("-"))
    num_days = calendar.monthrange(year, m)[1]
    month_start = f"{month}-01"
    month_end = f"{month}-{num_days:02d}"

    match_stage: Dict[str, Any] = {"joined_on": {"$lte": month_end}}
    if department:
        match_stage["department"] = department

    pipeline = [
        {"$match": match_stage},
        {
            "$lookup": {
                "from": "attendance_logs",
                "let": {"code": "$emp_code"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$eq": ["$emp_code", "$$code"]},
                                    {"$gte": ["$date", month_start]},
                                    {"$lte": ["$date", month_end]},
                                ]
                            }
                        }
                    }
                ],
                "as": "logs",
            }
        },
        {
            "$unwind": {
                "path": "$logs",
                "preserveNullAndEmptyArrays": True,
            }
        },
        {
            "$group": {
                "_id": "$department",
                "employees": {"$addToSet": "$emp_code"},
                "present_days": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$ne": ["$logs", None]},
                                    {"$in": ["$logs.status", ["PRESENT", "WFH", "ON_DUTY"]]},
                                    {
                                        "$in": [
                                            {"$dayOfWeek": {"$dateFromString": {"dateString": "$logs.date"}}},
                                            [2, 3, 4, 5, 6],
                                        ]
                                    },
                                ]
                            },
                            {"$cond": ["$logs.half_day", 0.5, 1.0]},
                            0.0,
                        ]
                    }
                },
                "work_hours_sum": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$ne": ["$logs", None]},
                                    {"$in": ["$logs.status", ["PRESENT", "WFH", "ON_DUTY"]]},
                                    {"$ne": ["$logs.work_hours", None]},
                                ]
                            },
                            "$logs.work_hours",
                            0.0,
                        ]
                    }
                },
                "work_hours_count": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$ne": ["$logs", None]},
                                    {"$in": ["$logs.status", ["PRESENT", "WFH", "ON_DUTY"]]},
                                    {"$ne": ["$logs.work_hours", None]},
                                ]
                            },
                            1,
                            0,
                        ]
                    }
                },
                "late_count": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$ne": ["$logs", None]},
                                    {"$gt": ["$logs.late_minutes", 0]},
                                ]
                            },
                            1,
                            0,
                        ]
                    }
                },
                "total_late_minutes": {
                    "$sum": {
                        "$cond": [
                            {"$ne": ["$logs", None]},
                            {"$ifNull": ["$logs.late_minutes", 0]},
                            0,
                        ]
                    }
                },
                "leave_count": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$ne": ["$logs", None]},
                                    {"$eq": ["$logs.status", "LEAVE"]},
                                ]
                            },
                            1,
                            0,
                        ]
                    }
                },
                "on_duty_count": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$ne": ["$logs", None]},
                                    {"$eq": ["$logs.status", "ON_DUTY"]},
                                ]
                            },
                            1,
                            0,
                        ]
                    }
                },
            }
        },
        {
            "$project": {
                "_id": 0,
                "department": "$_id",
                "headcount": {"$size": "$employees"},
                "present_days": {"$round": ["$present_days", 2]},
                "avg_work_hours": {
                    "$cond": [
                        {"$gt": ["$work_hours_count", 0]},
                        {"$round": [{"$divide": ["$work_hours_sum", "$work_hours_count"]}, 2]},
                        None,
                    ]
                },
                "late_count": 1,
                "total_late_minutes": 1,
                "leave_count": 1,
                "on_duty_count": 1,
            }
        },
        {"$sort": {"department": 1}},
    ]

    items = list(db.employees.aggregate(pipeline))
    return {"month": month, "items": items}


@app.get("/analytics/leaderboard/late")
def late_leaderboard(
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    limit: int = Query(10, ge=1, le=50),
    department: Optional[str] = Query(None),
):
    year, m = map(int, month.split("-"))
    num_days = calendar.monthrange(year, m)[1]
    month_start = f"{month}-01"
    month_end = f"{month}-{num_days:02d}"

    pipeline = [
        {
            "$match": {
                "date": {"$gte": month_start, "$lte": month_end},
                "late_minutes": {"$gt": 0},
            }
        },
        {
            "$group": {
                "_id": "$emp_code",
                "total_late_minutes": {"$sum": "$late_minutes"},
                "late_count": {"$sum": 1},
            }
        },
        {
            "$lookup": {
                "from": "employees",
                "localField": "_id",
                "foreignField": "emp_code",
                "as": "emp",
            }
        },
        {"$unwind": "$emp"},
    ]

    if department:
        pipeline.append({"$match": {"emp.department": department}})

    pipeline.extend([
        {
            "$setWindowFields": {
                "sortBy": {"total_late_minutes": -1},
                "output": {
                    "rank": {"$rank": {}}
                },
            }
        },
        {"$match": {"rank": {"$lte": limit}}},
        {"$sort": {"total_late_minutes": -1, "_id": 1}},
        {
            "$project": {
                "_id": 0,
                "rank": 1,
                "emp_code": "$_id",
                "name": "$emp.name",
                "department": "$emp.department",
                "total_late_minutes": 1,
                "late_count": 1,
            }
        },
    ])

    items = list(db.attendance_logs.aggregate(pipeline))
    return {"month": month, "items": items}


@app.get("/analytics/departments/{department}/trend")
def department_trend(
    department: str = Path(...),
    from_date: str = Query(..., alias="from"),
    to_date: str = Query(..., alias="to"),
):
    try:
        from_d = date.fromisoformat(from_date)
        to_d = date.fromisoformat(to_date)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid date format, expected YYYY-MM-DD")

    if to_d < from_d:
        raise HTTPException(status_code=422, detail="'to' must be >= 'from'")
    # Range is inclusive calendar days
    if (to_d - from_d).days + 1 > 92:
        raise HTTPException(status_code=422, detail="Date range exceeds 92 days")

    # 404 for unknown department (no employees)
    if db.employees.count_documents({"department": department}) == 0:
        raise HTTPException(status_code=404, detail="Department not found")

    from_dt = datetime(from_d.year, from_d.month, from_d.day, tzinfo=timezone.utc)
    to_dt = datetime(to_d.year, to_d.month, to_d.day, tzinfo=timezone.utc)
    to_dt_next = to_dt + timedelta(days=1)

    emp_codes = [e["emp_code"] for e in db.employees.find({"department": department}, {"emp_code": 1})]

    pipeline = [
        {
            "$match": {
                "emp_code": {"$in": emp_codes},
                "date": {"$gte": from_date, "$lte": to_date},
            }
        },
        {
            "$project": {
                "date": 1,
                "status": 1,
                "late_minutes": 1,
                "half_day": 1,
                "date_dt": {"$dateFromString": {"dateString": "$date"}},
            }
        },
        {
            "$unionWith": {
                "pipeline": [
                    {
                        "$documents": [
                            {"date_dt": from_dt, "is_dummy": True},
                            {"date_dt": to_dt, "is_dummy": True},
                        ]
                    }
                ]
            }
        },
        {
            "$group": {
                "_id": "$date_dt",
                "present_count": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$ne": ["$is_dummy", True]},
                                    {"$in": ["$status", ["PRESENT", "WFH", "ON_DUTY"]]},
                                ]
                            },
                            {"$cond": ["$half_day", 0.5, 1.0]},
                            0.0,
                        ]
                    }
                },
                "late_count": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$ne": ["$is_dummy", True]},
                                    {"$gt": ["$late_minutes", 0]},
                                ]
                            },
                            1,
                            0,
                        ]
                    }
                },
            }
        },
        {
            "$project": {
                "_id": 0,
                "date_dt": "$_id",
                "present_count": 1,
                "late_count": 1,
            }
        },
        {
            "$densify": {
                "field": "date_dt",
                "range": {
                    "step": 1,
                    "unit": "day",
                    "bounds": [from_dt, to_dt_next],
                },
            }
        },
        {"$sort": {"date_dt": 1}},
        {
            "$set": {
                "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$date_dt"}},
                "present_count": {"$ifNull": ["$present_count", 0.0]},
                "late_count": {"$ifNull": ["$late_count", 0]},
                "is_working_day": {
                    "$in": [
                        {"$dayOfWeek": "$date_dt"},
                        [2, 3, 4, 5, 6],
                    ]
                },
            }
        },
        {
            "$lookup": {
                "from": "employees",
                "let": {"cur_date": "$date"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$eq": ["$department", department]},
                                    {"$lte": ["$joined_on", "$$cur_date"]},
                                ]
                            }
                        }
                    },
                    {"$count": "total"},
                ],
                "as": "hc",
            }
        },
        {
            "$set": {
                "headcount": {"$ifNull": [{"$first": "$hc.total"}, 0]},
            }
        },
        {
            "$set": {
                "attendance_rate": {
                    "$cond": [
                        {
                            "$and": [
                                "$is_working_day",
                                {"$gt": ["$headcount", 0]},
                            ]
                        },
                        {"$round": [{"$divide": ["$present_count", "$headcount"]}, 4]},
                        None,
                    ]
                }
            }
        },
        {
            "$setWindowFields": {
                "sortBy": {"date_dt": 1},
                "output": {
                    "raw_moving_avg": {
                        "$avg": "$attendance_rate",
                        "window": {
                            "documents": [-6, "current"],
                        },
                    }
                },
            }
        },
        {
            "$project": {
                "_id": 0,
                "date": 1,
                "is_working_day": 1,
                "headcount": 1,
                "present_count": {"$round": ["$present_count", 2]},
                "late_count": 1,
                "attendance_rate": 1,
                "moving_avg_7d": {
                    "$cond": [
                        {"$ne": ["$raw_moving_avg", None]},
                        {"$round": ["$raw_moving_avg", 4]},
                        None,
                    ]
                },
            }
        },
    ]

    items = list(db.attendance_logs.aggregate(pipeline))
    return {"department": department, "items": items}


# --------------------------------------------------------------------------- #
# Admin Explain Endpoint
# --------------------------------------------------------------------------- #
@app.get("/admin/explain/{endpoint}")
def explain_endpoint(
    endpoint: str = Path(..., pattern=r"^(attendance_list|employee_monthly|department_summary|late_leaderboard|department_trend)$"),
    emp_code: Optional[str] = Query(None),
    month: Optional[str] = Query(None, pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    department: Optional[str] = Query(None),
    limit: Optional[int] = Query(10, ge=1, le=50),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    status_filter: Optional[str] = Query(None, alias="status", pattern=r"^(PRESENT|ABSENT|LEAVE|WFH|ON_DUTY)$"),
    from_date: Optional[str] = Query(None, alias="from"),
    to_date: Optional[str] = Query(None, alias="to"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    if endpoint == "attendance_list":
        if date_from and date_to and date_from > date_to:
            raise HTTPException(status_code=422, detail="date_from must be <= date_to")

        q: Dict[str, Any] = {}
        if emp_code:
            q["emp_code"] = emp_code
        if date_from or date_to:
            q["date"] = {}
            if date_from:
                q["date"]["$gte"] = date_from
            if date_to:
                q["date"]["$lte"] = date_to
        if status_filter:
            q["status"] = status_filter

        skip = (page - 1) * page_size
        exp = db.command("explain", {
            "find": "attendance_logs",
            "filter": q,
            "sort": SON([("date", -1), ("emp_code", 1)]),
            "skip": skip,
            "limit": page_size,
        }, verbosity="executionStats")
        collection = "attendance_logs"

    elif endpoint == "employee_monthly":
        if not emp_code:
            raise HTTPException(status_code=422, detail="emp_code is required for employee_monthly")
        if not month:
            raise HTTPException(status_code=422, detail="month is required for employee_monthly")

        year, m = map(int, month.split("-"))
        num_days = calendar.monthrange(year, m)[1]
        month_start = f"{month}-01"
        month_end = f"{month}-{num_days:02d}"

        pipeline = [
            {
                "$match": {
                    "emp_code": emp_code,
                    "date": {"$gte": month_start, "$lte": month_end},
                }
            },
            {
                "$project": {
                    "status": 1,
                    "date": 1,
                    "half_day": {"$ifNull": ["$half_day", False]},
                    "late_minutes": {"$ifNull": ["$late_minutes", 0]},
                    "overtime_minutes": {"$ifNull": ["$overtime_minutes", 0]},
                    "is_weekday": {
                        "$in": [
                            {"$dayOfWeek": {"$dateFromString": {"dateString": "$date"}}},
                            [2, 3, 4, 5, 6],
                        ]
                    },
                    "is_present": {"$in": ["$status", ["PRESENT", "WFH", "ON_DUTY"]]},
                }
            },
            {
                "$group": {
                    "_id": "$emp_code",
                    "present_days": {
                        "$sum": {
                            "$cond": [
                                {"$and": ["$is_weekday", "$is_present"]},
                                {"$cond": ["$half_day", 0.5, 1.0]},
                                0.0,
                            ]
                        }
                    },
                    "leave_days": {
                        "$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}
                    },
                    "late_count": {
                        "$sum": {"$cond": [{"$gt": ["$late_minutes", 0]}, 1, 0]}
                    },
                    "total_late_minutes": {"$sum": "$late_minutes"},
                    "total_overtime_minutes": {"$sum": "$overtime_minutes"},
                }
            },
        ]
        exp = db.command("explain", {
            "aggregate": "attendance_logs",
            "pipeline": pipeline,
            "cursor": {},
        }, verbosity="executionStats")
        collection = "attendance_logs"

    elif endpoint == "department_summary":
        if not month:
            raise HTTPException(status_code=422, detail="month is required for department_summary")

        year, m = map(int, month.split("-"))
        num_days = calendar.monthrange(year, m)[1]
        month_start = f"{month}-01"
        month_end = f"{month}-{num_days:02d}"

        match_stage: Dict[str, Any] = {"joined_on": {"$lte": month_end}}
        if department:
            match_stage["department"] = department

        pipeline = [
            {"$match": match_stage},
            {
                "$lookup": {
                    "from": "attendance_logs",
                    "let": {"code": "$emp_code"},
                    "pipeline": [
                        {
                            "$match": {
                                "$expr": {
                                    "$and": [
                                        {"$eq": ["$emp_code", "$$code"]},
                                        {"$gte": ["$date", month_start]},
                                        {"$lte": ["$date", month_end]},
                                    ]
                                }
                            }
                        }
                    ],
                    "as": "logs",
                }
            },
            {
                "$unwind": {
                    "path": "$logs",
                    "preserveNullAndEmptyArrays": True,
                }
            },
            {
                "$group": {
                    "_id": "$department",
                    "employees": {"$addToSet": "$emp_code"},
                    "present_days": {
                        "$sum": {
                            "$cond": [
                                {
                                    "$and": [
                                        {"$ne": ["$logs", None]},
                                        {"$in": ["$logs.status", ["PRESENT", "WFH", "ON_DUTY"]]},
                                        {
                                            "$in": [
                                                {"$dayOfWeek": {"$dateFromString": {"dateString": "$logs.date"}}},
                                                [2, 3, 4, 5, 6],
                                            ]
                                        },
                                    ]
                                },
                                {"$cond": ["$logs.half_day", 0.5, 1.0]},
                                0.0,
                            ]
                        }
                    },
                    "work_hours_sum": {
                        "$sum": {
                            "$cond": [
                                {
                                    "$and": [
                                        {"$ne": ["$logs", None]},
                                        {"$in": ["$logs.status", ["PRESENT", "WFH", "ON_DUTY"]]},
                                        {"$ne": ["$logs.work_hours", None]},
                                    ]
                                },
                                "$logs.work_hours",
                                0.0,
                            ]
                        }
                    },
                    "work_hours_count": {
                        "$sum": {
                            "$cond": [
                                {
                                    "$and": [
                                        {"$ne": ["$logs", None]},
                                        {"$in": ["$logs.status", ["PRESENT", "WFH", "ON_DUTY"]]},
                                        {"$ne": ["$logs.work_hours", None]},
                                    ]
                                },
                                1,
                                0,
                            ]
                        }
                    },
                    "late_count": {
                        "$sum": {
                            "$cond": [
                                {
                                    "$and": [
                                        {"$ne": ["$logs", None]},
                                        {"$gt": ["$logs.late_minutes", 0]},
                                    ]
                                },
                                1,
                                0,
                            ]
                        }
                    },
                    "total_late_minutes": {
                        "$sum": {
                            "$cond": [
                                {"$ne": ["$logs", None]},
                                {"$ifNull": ["$logs.late_minutes", 0]},
                                0,
                            ]
                        }
                    },
                    "leave_count": {
                        "$sum": {
                            "$cond": [
                                {
                                    "$and": [
                                        {"$ne": ["$logs", None]},
                                        {"$eq": ["$logs.status", "LEAVE"]},
                                    ]
                                },
                                1,
                                0,
                            ]
                        }
                    },
                    "on_duty_count": {
                        "$sum": {
                            "$cond": [
                                {
                                    "$and": [
                                        {"$ne": ["$logs", None]},
                                        {"$eq": ["$logs.status", "ON_DUTY"]},
                                    ]
                                },
                                1,
                                0,
                            ]
                        }
                    },
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "department": "$_id",
                    "headcount": {"$size": "$employees"},
                    "present_days": {"$round": ["$present_days", 2]},
                    "avg_work_hours": {
                        "$cond": [
                            {"$gt": ["$work_hours_count", 0]},
                            {"$round": [{"$divide": ["$work_hours_sum", "$work_hours_count"]}, 2]},
                            None,
                        ]
                    },
                    "late_count": 1,
                    "total_late_minutes": 1,
                    "leave_count": 1,
                    "on_duty_count": 1,
                }
            },
            {"$sort": {"department": 1}},
        ]
        exp = db.command("explain", {
            "aggregate": "employees",
            "pipeline": pipeline,
            "cursor": {},
        }, verbosity="executionStats")
        collection = "employees"

    elif endpoint == "late_leaderboard":
        if not month:
            raise HTTPException(status_code=422, detail="month is required for late_leaderboard")

        year, m = map(int, month.split("-"))
        num_days = calendar.monthrange(year, m)[1]
        month_start = f"{month}-01"
        month_end = f"{month}-{num_days:02d}"

        pipeline = [
            {
                "$match": {
                    "date": {"$gte": month_start, "$lte": month_end},
                    "late_minutes": {"$gt": 0},
                }
            },
            {
                "$group": {
                    "_id": "$emp_code",
                    "total_late_minutes": {"$sum": "$late_minutes"},
                    "late_count": {"$sum": 1},
                }
            },
            {
                "$lookup": {
                    "from": "employees",
                    "localField": "_id",
                    "foreignField": "emp_code",
                    "as": "emp",
                }
            },
            {"$unwind": "$emp"},
        ]

        if department:
            pipeline.append({"$match": {"emp.department": department}})

        pipeline.extend([
            {
                "$setWindowFields": {
                    "sortBy": {"total_late_minutes": -1},
                    "output": {
                        "rank": {"$rank": {}}
                    },
                }
            },
            {"$match": {"rank": {"$lte": limit if limit is not None else 10}}},
            {"$sort": {"total_late_minutes": -1, "_id": 1}},
            {
                "$project": {
                    "_id": 0,
                    "rank": 1,
                    "emp_code": "$_id",
                    "name": "$emp.name",
                    "department": "$emp.department",
                    "total_late_minutes": 1,
                    "late_count": 1,
                }
            },
        ])
        exp = db.command("explain", {
            "aggregate": "attendance_logs",
            "pipeline": pipeline,
            "cursor": {},
        }, verbosity="executionStats")
        collection = "attendance_logs"

    elif endpoint == "department_trend":
        if not department:
            raise HTTPException(status_code=422, detail="department is required for department_trend")
        if not from_date or not to_date:
            raise HTTPException(status_code=422, detail="'from' and 'to' are required for department_trend")

        try:
            from_d = date.fromisoformat(from_date)
            to_d = date.fromisoformat(to_date)
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid date format, expected YYYY-MM-DD")

        if to_d < from_d:
            raise HTTPException(status_code=422, detail="'to' must be >= 'from'")
        if (to_d - from_d).days + 1 > 92:
            raise HTTPException(status_code=422, detail="Date range exceeds 92 days")

        from_dt = datetime(from_d.year, from_d.month, from_d.day, tzinfo=timezone.utc)
        to_dt = datetime(to_d.year, to_d.month, to_d.day, tzinfo=timezone.utc)
        to_dt_next = to_dt + timedelta(days=1)

        emp_codes = [e["emp_code"] for e in db.employees.find({"department": department}, {"emp_code": 1})]

        pipeline = [
            {
                "$match": {
                    "emp_code": {"$in": emp_codes},
                    "date": {"$gte": from_date, "$lte": to_date},
                }
            },
            {
                "$project": {
                    "date": 1,
                    "status": 1,
                    "late_minutes": 1,
                    "half_day": 1,
                    "date_dt": {"$dateFromString": {"dateString": "$date"}},
                }
            },
            {
                "$unionWith": {
                    "pipeline": [
                        {
                            "$documents": [
                                {"date_dt": from_dt, "is_dummy": True},
                                {"date_dt": to_dt, "is_dummy": True},
                            ]
                        }
                    ]
                }
            },
            {
                "$group": {
                    "_id": "$date_dt",
                    "present_count": {
                        "$sum": {
                            "$cond": [
                                {
                                    "$and": [
                                        {"$ne": ["$is_dummy", True]},
                                        {"$in": ["$status", ["PRESENT", "WFH", "ON_DUTY"]]},
                                    ]
                                },
                                {"$cond": ["$half_day", 0.5, 1.0]},
                                0.0,
                            ]
                        }
                    },
                    "late_count": {
                        "$sum": {
                            "$cond": [
                                {
                                    "$and": [
                                        {"$ne": ["$is_dummy", True]},
                                        {"$gt": ["$late_minutes", 0]},
                                    ]
                                },
                                1,
                                0,
                            ]
                        }
                    },
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "date_dt": "$_id",
                    "present_count": 1,
                    "late_count": 1,
                }
            },
            {
                "$densify": {
                    "field": "date_dt",
                    "range": {
                        "step": 1,
                        "unit": "day",
                        "bounds": [from_dt, to_dt_next],
                    },
                }
            },
            {"$sort": {"date_dt": 1}},
            {
                "$set": {
                    "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$date_dt"}},
                    "present_count": {"$ifNull": ["$present_count", 0.0]},
                    "late_count": {"$ifNull": ["$late_count", 0]},
                    "is_working_day": {
                        "$in": [
                            {"$dayOfWeek": "$date_dt"},
                            [2, 3, 4, 5, 6],
                        ]
                    },
                }
            },
            {
                "$lookup": {
                    "from": "employees",
                    "let": {"cur_date": "$date"},
                    "pipeline": [
                        {
                            "$match": {
                                "$expr": {
                                    "$and": [
                                        {"$eq": ["$department", department]},
                                        {"$lte": ["$joined_on", "$$cur_date"]},
                                    ]
                                }
                            }
                        },
                        {"$count": "total"},
                    ],
                    "as": "hc",
                }
            },
            {
                "$set": {
                    "headcount": {"$ifNull": [{"$first": "$hc.total"}, 0]},
                }
            },
            {
                "$set": {
                    "attendance_rate": {
                        "$cond": [
                            {
                                "$and": [
                                    "$is_working_day",
                                    {"$gt": ["$headcount", 0]},
                                ]
                            },
                            {"$round": [{"$divide": ["$present_count", "$headcount"]}, 4]},
                            None,
                        ]
                    }
                }
            },
            {
                "$setWindowFields": {
                    "sortBy": {"date_dt": 1},
                    "output": {
                        "raw_moving_avg": {
                            "$avg": "$attendance_rate",
                            "window": {
                                "documents": [-6, "current"],
                            },
                        }
                    },
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "date": 1,
                    "is_working_day": 1,
                    "headcount": 1,
                    "present_count": {"$round": ["$present_count", 2]},
                    "late_count": 1,
                    "attendance_rate": 1,
                    "moving_avg_7d": {
                        "$cond": [
                            {"$ne": ["$raw_moving_avg", None]},
                            {"$round": ["$raw_moving_avg", 4]},
                            None,
                        ]
                    },
                }
            },
        ]
        exp = db.command("explain", {
            "aggregate": "attendance_logs",
            "pipeline": pipeline,
            "cursor": {},
        }, verbosity="executionStats")
        collection = "attendance_logs"

    else:
        raise HTTPException(status_code=422, detail=f"Unknown endpoint '{endpoint}'")

    raw_explain = json.loads(json_util.dumps(exp))
    return {
        "endpoint": endpoint,
        "collection": collection,
        "explain": raw_explain,
    }


# --------------------------------------------------------------------------- #
# Frontend Web Dashboard (Served at /)
# --------------------------------------------------------------------------- #
FRONTEND_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>HROne Attendance & Analytics System</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    body { font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif; }
    .tab-content { display: none; }
    .tab-content.active { display: block; }
  </style>
</head>
<body class="bg-slate-50 text-slate-800 min-h-screen">
  <!-- Navbar -->
  <header class="bg-indigo-700 text-white shadow-md sticky top-0 z-40">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex justify-between items-center h-16">
      <div class="flex items-center space-x-3">
        <div class="bg-white text-indigo-700 p-2 rounded-lg font-black text-xl shadow">
          <i class="fa-solid fa-clock-rotate-left"></i>
        </div>
        <div>
          <h1 class="text-xl font-bold tracking-tight">HROne Attendance & Analytics</h1>
          <p class="text-xs text-indigo-200">FastAPI &amp; MongoDB System</p>
        </div>
      </div>
      <div class="flex items-center space-x-4">
        <div id="health-badge" class="flex items-center text-xs font-semibold px-3 py-1.5 rounded-full bg-indigo-800 border border-indigo-600">
          <span class="w-2 h-2 rounded-full bg-yellow-400 mr-2 animate-pulse" id="health-dot"></span>
          <span id="health-text">Checking MongoDB...</span>
        </div>
        <a href="/docs" target="_blank" class="text-xs bg-indigo-600 hover:bg-indigo-500 text-white font-medium px-3 py-1.5 rounded-lg border border-indigo-400 transition flex items-center">
          <i class="fa-solid fa-book-open mr-1.5"></i> Swagger API Docs
        </a>
      </div>
    </div>

    <!-- Navigation Tabs -->
    <div class="bg-indigo-800 border-t border-indigo-600 px-4 sm:px-8">
      <nav class="flex space-x-1 sm:space-x-4 overflow-x-auto py-2 text-sm font-medium">
        <button onclick="switchTab('punch')" id="tab-btn-punch" class="tab-btn px-3 py-2 rounded-md transition text-white bg-indigo-900 shadow-inner flex items-center">
          <i class="fa-solid fa-fingerprint mr-2"></i> Punch Clock
        </button>
        <button onclick="switchTab('attendance')" id="tab-btn-attendance" class="tab-btn px-3 py-2 rounded-md transition text-indigo-200 hover:text-white hover:bg-indigo-700 flex items-center">
          <i class="fa-solid fa-calendar-check mr-2"></i> Attendance Records
        </button>
        <button onclick="switchTab('analytics')" id="tab-btn-analytics" class="tab-btn px-3 py-2 rounded-md transition text-indigo-200 hover:text-white hover:bg-indigo-700 flex items-center">
          <i class="fa-solid fa-chart-line mr-2"></i> Analytics &amp; Reports
        </button>
        <button onclick="switchTab('employees')" id="tab-btn-employees" class="tab-btn px-3 py-2 rounded-md transition text-indigo-200 hover:text-white hover:bg-indigo-700 flex items-center">
          <i class="fa-solid fa-users mr-2"></i> Employees Directory
        </button>
      </nav>
    </div>
  </header>

  <!-- Main Container -->
  <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
    
    <!-- Tab 1: Punch Clock -->
    <div id="tab-punch" class="tab-content active space-y-6">
      <div class="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
        <div class="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-slate-100">
          <div>
            <h2 class="text-xl font-bold text-slate-800 flex items-center">
              <i class="fa-solid fa-stopwatch text-indigo-600 mr-2.5"></i> Employee Punch Terminal
            </h2>
            <p class="text-sm text-slate-500 mt-1">Record punch-in (R1, R2) and punch-out (R3, R4, R5) with real-time rule calculation.</p>
          </div>
          <div class="bg-slate-50 border border-slate-200 rounded-xl px-4 py-2 text-right">
            <span class="text-xs text-slate-400 font-semibold uppercase tracking-wider block">Current IST Time</span>
            <span id="live-clock" class="text-lg font-mono font-bold text-indigo-700">--:--:--</span>
          </div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-8 mt-6">
          <!-- Punch In Card -->
          <div class="bg-slate-50 rounded-xl p-6 border border-slate-200 flex flex-col justify-between">
            <div>
              <div class="flex items-center justify-between mb-4">
                <span class="text-sm font-bold uppercase tracking-wider text-emerald-700 flex items-center">
                  <i class="fa-solid fa-right-to-bracket mr-2"></i> Punch-In
                </span>
                <span class="text-xs bg-emerald-100 text-emerald-800 font-semibold px-2 py-0.5 rounded">R1 &amp; R2 Rules</span>
              </div>
              <div class="space-y-4">
                <div>
                  <label class="block text-xs font-semibold text-slate-600 uppercase mb-1">Select Employee</label>
                  <select id="punchin-emp" class="w-full bg-white border border-slate-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                    <option value="">Loading employees...</option>
                  </select>
                </div>
                <div>
                  <label class="block text-xs font-semibold text-slate-600 uppercase mb-1">Presence Status</label>
                  <select id="punchin-status" class="w-full bg-white border border-slate-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                    <option value="PRESENT">PRESENT (On-site)</option>
                    <option value="WFH">WFH (Work From Home)</option>
                    <option value="ON_DUTY">ON_DUTY (Official Duty)</option>
                  </select>
                </div>
                <div class="bg-white p-3 rounded-lg border border-slate-200">
                  <div class="flex items-center justify-between text-xs">
                    <span class="font-semibold text-slate-600">Punch Time:</span>
                    <span class="inline-flex items-center text-emerald-700 font-bold">
                      <span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse mr-1.5"></span>
                      Current Time (Automatic)
                    </span>
                  </div>
                  <p class="text-[11px] text-slate-400 mt-1">Automatically recorded from the current instant in IST.</p>
                </div>
              </div>
            </div>
            <button onclick="submitPunchIn()" class="mt-6 w-full bg-emerald-600 hover:bg-emerald-700 text-white font-semibold py-2.5 px-4 rounded-lg shadow transition flex items-center justify-center">
              <i class="fa-solid fa-arrow-right-to-bracket mr-2"></i> Punch-In Now
            </button>
          </div>

          <!-- Punch Out Card -->
          <div class="bg-slate-50 rounded-xl p-6 border border-slate-200 flex flex-col justify-between">
            <div>
              <div class="flex items-center justify-between mb-4">
                <span class="text-sm font-bold uppercase tracking-wider text-amber-700 flex items-center">
                  <i class="fa-solid fa-right-from-bracket mr-2"></i> Punch-Out
                </span>
                <span class="text-xs bg-amber-100 text-amber-800 font-semibold px-2 py-0.5 rounded">R3 - R5 Rules</span>
              </div>
              <div class="space-y-4">
                <div>
                  <label class="block text-xs font-semibold text-slate-600 uppercase mb-1">Select Employee</label>
                  <select id="punchout-emp" class="w-full bg-white border border-slate-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                    <option value="">Loading employees...</option>
                  </select>
                </div>
                <div class="bg-white p-3 rounded-lg border border-slate-200">
                  <div class="flex items-center justify-between text-xs">
                    <span class="font-semibold text-slate-600">Punch Time:</span>
                    <span class="inline-flex items-center text-amber-700 font-bold">
                      <span class="w-2 h-2 rounded-full bg-amber-500 animate-pulse mr-1.5"></span>
                      Current Time (Automatic)
                    </span>
                  </div>
                  <p class="text-[11px] text-slate-400 mt-1">Automatically closes the open shift at current instant.</p>
                </div>
              </div>
            </div>
            <button onclick="submitPunchOut()" class="mt-6 w-full bg-amber-600 hover:bg-amber-700 text-white font-semibold py-2.5 px-4 rounded-lg shadow transition flex items-center justify-center">
              <i class="fa-solid fa-arrow-right-from-bracket mr-2"></i> Punch-Out Now
            </button>
          </div>
        </div>

        <!-- Punch Result Box -->
        <div id="punch-result" class="hidden mt-6 p-4 rounded-xl border text-sm"></div>
      </div>
    </div>

    <!-- Tab 2: Attendance Records -->
    <div id="tab-attendance" class="tab-content space-y-6">
      <div class="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
        <div class="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-6">
          <div>
            <h2 class="text-xl font-bold text-slate-800 flex items-center">
              <i class="fa-solid fa-clipboard-user text-indigo-600 mr-2.5"></i> Daily Attendance Records
            </h2>
            <p class="text-sm text-slate-500 mt-1">Paginated index-backed query sorted by date descending, emp_code ascending.</p>
          </div>
          <button onclick="loadAttendanceLogs(1)" class="text-xs bg-slate-100 hover:bg-slate-200 text-slate-700 font-semibold px-3 py-2 rounded-lg border border-slate-300 flex items-center transition">
            <i class="fa-solid fa-arrows-rotate mr-1.5"></i> Refresh
          </button>
        </div>

        <!-- Filters -->
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4 bg-slate-50 p-4 rounded-xl border border-slate-200 mb-6">
          <div>
            <label class="block text-xs font-semibold text-slate-600 mb-1">Employee</label>
            <input type="text" id="att-filter-emp" placeholder="e.g. EMP0001" class="w-full bg-white border border-slate-300 rounded-lg px-3 py-1.5 text-xs focus:ring-2 focus:ring-indigo-500 focus:outline-none">
          </div>
          <div>
            <label class="block text-xs font-semibold text-slate-600 mb-1">Date From</label>
            <input type="date" id="att-filter-from" class="w-full bg-white border border-slate-300 rounded-lg px-3 py-1.5 text-xs focus:ring-2 focus:ring-indigo-500 focus:outline-none">
          </div>
          <div>
            <label class="block text-xs font-semibold text-slate-600 mb-1">Date To</label>
            <input type="date" id="att-filter-to" class="w-full bg-white border border-slate-300 rounded-lg px-3 py-1.5 text-xs focus:ring-2 focus:ring-indigo-500 focus:outline-none">
          </div>
          <div>
            <label class="block text-xs font-semibold text-slate-600 mb-1">Status</label>
            <select id="att-filter-status" class="w-full bg-white border border-slate-300 rounded-lg px-3 py-1.5 text-xs focus:ring-2 focus:ring-indigo-500 focus:outline-none">
              <option value="">All Statuses</option>
              <option value="PRESENT">PRESENT</option>
              <option value="WFH">WFH</option>
              <option value="ON_DUTY">ON_DUTY</option>
              <option value="ABSENT">ABSENT</option>
              <option value="LEAVE">LEAVE</option>
            </select>
          </div>
          <div class="flex items-end">
            <button onclick="loadAttendanceLogs(1)" class="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-1.5 px-3 rounded-lg text-xs transition">
              Apply Filters
            </button>
          </div>
        </div>

        <!-- Table -->
        <div class="overflow-x-auto rounded-xl border border-slate-200">
          <table class="min-w-full divide-y divide-slate-200 text-left text-xs">
            <thead class="bg-slate-100 font-semibold text-slate-700">
              <tr>
                <th class="px-4 py-3">Date</th>
                <th class="px-4 py-3">Employee</th>
                <th class="px-4 py-3">Status</th>
                <th class="px-4 py-3">Punch-In (IST)</th>
                <th class="px-4 py-3">Punch-Out (IST)</th>
                <th class="px-4 py-3">Work Hours</th>
                <th class="px-4 py-3">Late Mins</th>
                <th class="px-4 py-3">Overtime</th>
                <th class="px-4 py-3">Half Day</th>
                <th class="px-4 py-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody id="attendance-tbody" class="divide-y divide-slate-100 bg-white">
              <tr><td colspan="10" class="text-center py-6 text-slate-400">Loading attendance logs...</td></tr>
            </tbody>
          </table>
        </div>

        <!-- Pagination Controls -->
        <div class="flex items-center justify-between mt-4 text-xs text-slate-500">
          <div id="att-pagination-info">Showing page 1</div>
          <div class="flex space-x-2">
            <button id="att-prev-btn" onclick="prevAttPage()" class="px-3 py-1.5 bg-slate-100 hover:bg-slate-200 border border-slate-300 rounded font-medium disabled:opacity-40">Previous</button>
            <button id="att-next-btn" onclick="nextAttPage()" class="px-3 py-1.5 bg-slate-100 hover:bg-slate-200 border border-slate-300 rounded font-medium disabled:opacity-40">Next</button>
          </div>
        </div>
      </div>
    </div>

    <!-- Tab 3: Analytics & Reports -->
    <div id="tab-analytics" class="tab-content space-y-6">
      <div class="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
        <div class="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-6 pb-4 border-b border-slate-100">
          <div>
            <h2 class="text-xl font-bold text-slate-800 flex items-center">
              <i class="fa-solid fa-chart-pie text-indigo-600 mr-2.5"></i> Analytics Pipelines Dashboard
            </h2>
            <p class="text-sm text-slate-500 mt-1">MongoDB aggregation-backed reports (cross-collection, window ranking &amp; $densify trend).</p>
          </div>
          <!-- Analytics Sub-navigation -->
          <div class="flex flex-wrap gap-2 text-xs font-semibold">
            <button onclick="switchAnalyticsSubTab('dept-summary')" id="subtab-btn-dept-summary" class="subtab-btn px-3 py-1.5 rounded-lg bg-indigo-100 text-indigo-700">Department Summary</button>
            <button onclick="switchAnalyticsSubTab('leaderboard')" id="subtab-btn-leaderboard" class="subtab-btn px-3 py-1.5 rounded-lg bg-slate-100 text-slate-600 hover:bg-slate-200">Late Leaderboard</button>
            <button onclick="switchAnalyticsSubTab('trend')" id="subtab-btn-trend" class="subtab-btn px-3 py-1.5 rounded-lg bg-slate-100 text-slate-600 hover:bg-slate-200">Department Trend</button>
            <button onclick="switchAnalyticsSubTab('emp-monthly')" id="subtab-btn-emp-monthly" class="subtab-btn px-3 py-1.5 rounded-lg bg-slate-100 text-slate-600 hover:bg-slate-200">Employee Monthly</button>
          </div>
        </div>

        <!-- Subtab A: Department Summary -->
        <div id="subtab-dept-summary" class="analytics-subtab space-y-4">
          <div class="flex items-center space-x-3 bg-slate-50 p-3 rounded-xl border border-slate-200 text-xs">
            <label class="font-semibold text-slate-600">Month (YYYY-MM):</label>
            <input type="month" id="dept-summary-month" value="2026-07" class="bg-white border border-slate-300 rounded px-2 py-1">
            <button onclick="loadDeptSummary()" class="bg-indigo-600 hover:bg-indigo-700 text-white px-3 py-1 rounded font-semibold transition">Run Aggregation</button>
          </div>
          <div class="overflow-x-auto rounded-xl border border-slate-200">
            <table class="min-w-full divide-y divide-slate-200 text-left text-xs">
              <thead class="bg-slate-100 font-semibold text-slate-700">
                <tr>
                  <th class="px-4 py-3">Department</th>
                  <th class="px-4 py-3">Headcount (R9)</th>
                  <th class="px-4 py-3">Present Days</th>
                  <th class="px-4 py-3">Avg Work Hours</th>
                  <th class="px-4 py-3">Late Count</th>
                  <th class="px-4 py-3">Total Late Mins</th>
                  <th class="px-4 py-3">Leave Count</th>
                  <th class="px-4 py-3">On Duty</th>
                </tr>
              </thead>
              <tbody id="dept-summary-tbody" class="divide-y divide-slate-100 bg-white">
                <tr><td colspan="8" class="text-center py-4 text-slate-400">Click Run Aggregation to view</td></tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- Subtab B: Late Leaderboard -->
        <div id="subtab-leaderboard" class="analytics-subtab hidden space-y-4">
          <div class="flex items-center space-x-3 bg-slate-50 p-3 rounded-xl border border-slate-200 text-xs">
            <label class="font-semibold text-slate-600">Month:</label>
            <input type="month" id="leaderboard-month" value="2026-07" class="bg-white border border-slate-300 rounded px-2 py-1">
            <label class="font-semibold text-slate-600">Limit:</label>
            <input type="number" id="leaderboard-limit" value="10" min="1" max="50" class="w-16 bg-white border border-slate-300 rounded px-2 py-1">
            <button onclick="loadLeaderboard()" class="bg-indigo-600 hover:bg-indigo-700 text-white px-3 py-1 rounded font-semibold transition">Calculate Ranks</button>
          </div>
          <p class="text-xs text-slate-500 italic">Competition ranking ($rank): ties share a rank and next rank is skipped (1, 2, 2, 4). Limit applied after ranking.</p>
          <div class="overflow-x-auto rounded-xl border border-slate-200">
            <table class="min-w-full divide-y divide-slate-200 text-left text-xs">
              <thead class="bg-slate-100 font-semibold text-slate-700">
                <tr>
                  <th class="px-4 py-3">Rank</th>
                  <th class="px-4 py-3">Emp Code</th>
                  <th class="px-4 py-3">Name</th>
                  <th class="px-4 py-3">Department</th>
                  <th class="px-4 py-3">Total Late Minutes</th>
                  <th class="px-4 py-3">Late Occurrences</th>
                </tr>
              </thead>
              <tbody id="leaderboard-tbody" class="divide-y divide-slate-100 bg-white">
                <tr><td colspan="6" class="text-center py-4 text-slate-400">Click Calculate Ranks to view</td></tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- Subtab C: Department Trend -->
        <div id="subtab-trend" class="analytics-subtab hidden space-y-4">
          <div class="flex flex-wrap items-center gap-3 bg-slate-50 p-3 rounded-xl border border-slate-200 text-xs">
            <label class="font-semibold text-slate-600">Department:</label>
            <select id="trend-dept" class="bg-white border border-slate-300 rounded px-2 py-1">
              <option value="Engineering">Engineering</option>
              <option value="Sales">Sales</option>
              <option value="Support">Support</option>
            </select>
            <label class="font-semibold text-slate-600">From:</label>
            <input type="date" id="trend-from" value="2026-07-05" class="bg-white border border-slate-300 rounded px-2 py-1">
            <label class="font-semibold text-slate-600">To:</label>
            <input type="date" id="trend-to" value="2026-07-15" class="bg-white border border-slate-300 rounded px-2 py-1">
            <button onclick="loadTrend()" class="bg-indigo-600 hover:bg-indigo-700 text-white px-3 py-1 rounded font-semibold transition">Run $densify Trend</button>
          </div>
          <div class="overflow-x-auto rounded-xl border border-slate-200">
            <table class="min-w-full divide-y divide-slate-200 text-left text-xs">
              <thead class="bg-slate-100 font-semibold text-slate-700">
                <tr>
                  <th class="px-4 py-3">Date</th>
                  <th class="px-4 py-3">Working Day?</th>
                  <th class="px-4 py-3">Headcount</th>
                  <th class="px-4 py-3">Present Count</th>
                  <th class="px-4 py-3">Late Count</th>
                  <th class="px-4 py-3">Attendance Rate</th>
                  <th class="px-4 py-3">7-Day Moving Avg</th>
                </tr>
              </thead>
              <tbody id="trend-tbody" class="divide-y divide-slate-100 bg-white">
                <tr><td colspan="7" class="text-center py-4 text-slate-400">Click Run $densify Trend to view</td></tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- Subtab D: Employee Monthly -->
        <div id="subtab-emp-monthly" class="analytics-subtab hidden space-y-4">
          <div class="flex items-center space-x-3 bg-slate-50 p-3 rounded-xl border border-slate-200 text-xs">
            <label class="font-semibold text-slate-600">Employee:</label>
            <select id="monthly-emp" class="bg-white border border-slate-300 rounded px-2 py-1"></select>
            <label class="font-semibold text-slate-600">Month:</label>
            <input type="month" id="monthly-month" value="2026-07" class="bg-white border border-slate-300 rounded px-2 py-1">
            <button onclick="loadEmpMonthly()" class="bg-indigo-600 hover:bg-indigo-700 text-white px-3 py-1 rounded font-semibold transition">View Monthly Summary</button>
          </div>
          <div id="emp-monthly-cards" class="hidden grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div class="bg-slate-50 border border-slate-200 rounded-xl p-4">
              <span class="text-xs text-slate-500 uppercase font-semibold">Working Days (R7)</span>
              <p id="m-card-working" class="text-2xl font-bold text-slate-800 mt-1">0</p>
            </div>
            <div class="bg-slate-50 border border-slate-200 rounded-xl p-4">
              <span class="text-xs text-slate-500 uppercase font-semibold">Present Days</span>
              <p id="m-card-present" class="text-2xl font-bold text-emerald-600 mt-1">0</p>
            </div>
            <div class="bg-slate-50 border border-slate-200 rounded-xl p-4">
              <span class="text-xs text-slate-500 uppercase font-semibold">Attendance %</span>
              <p id="m-card-pct" class="text-2xl font-bold text-indigo-600 mt-1">0%</p>
            </div>
            <div class="bg-slate-50 border border-slate-200 rounded-xl p-4">
              <span class="text-xs text-slate-500 uppercase font-semibold">Late Minutes</span>
              <p id="m-card-late" class="text-2xl font-bold text-rose-600 mt-1">0</p>
            </div>
          </div>
        </div>

      </div>
    </div>

    <!-- Tab 4: Employee Directory -->
    <div id="tab-employees" class="tab-content space-y-6">
      <div class="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
        <div class="flex justify-between items-center mb-6">
          <div>
            <h2 class="text-xl font-bold text-slate-800 flex items-center">
              <i class="fa-solid fa-users text-indigo-600 mr-2.5"></i> Employee Directory
            </h2>
            <p class="text-sm text-slate-500 mt-1">Managed personnel with customer-assigned unique emp_code and configured shifts.</p>
          </div>
          <button onclick="openAddEmpModal()" class="bg-indigo-600 hover:bg-indigo-700 text-white font-semibold text-xs px-4 py-2 rounded-lg shadow transition flex items-center">
            <i class="fa-solid fa-user-plus mr-1.5"></i> Add Employee
          </button>
        </div>

        <div class="overflow-x-auto rounded-xl border border-slate-200">
          <table class="min-w-full divide-y divide-slate-200 text-left text-xs">
            <thead class="bg-slate-100 font-semibold text-slate-700">
              <tr>
                <th class="px-4 py-3">Emp Code</th>
                <th class="px-4 py-3">Name</th>
                <th class="px-4 py-3">Email</th>
                <th class="px-4 py-3">Department</th>
                <th class="px-4 py-3">Shift Window (IST)</th>
                <th class="px-4 py-3">Joined On</th>
              </tr>
            </thead>
            <tbody id="employees-tbody" class="divide-y divide-slate-100 bg-white">
              <tr><td colspan="6" class="text-center py-6 text-slate-400">Loading employees...</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

  </main>

  <!-- Modal: Regularize Attendance (Audit Trail) -->
  <div id="regularize-modal" class="hidden fixed inset-0 bg-slate-900/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
    <div class="bg-white rounded-2xl max-w-lg w-full p-6 shadow-2xl border border-slate-200 space-y-4">
      <div class="flex justify-between items-center pb-3 border-b border-slate-100">
        <h3 class="text-base font-bold text-slate-800 flex items-center">
          <i class="fa-solid fa-pen-to-square text-indigo-600 mr-2"></i> Regularize Attendance Record
        </h3>
        <button onclick="closeRegularizeModal()" class="text-slate-400 hover:text-slate-600 text-lg">&times;</button>
      </div>

      <div class="text-xs text-slate-500 bg-indigo-50 border border-indigo-100 rounded-lg p-2.5">
        Updating record for <strong id="reg-emp"></strong> on <strong id="reg-date"></strong>.
        Changes will be tracked in the immutable audit history.
      </div>

      <div class="space-y-3 text-xs">
        <div>
          <label class="block font-semibold text-slate-600 mb-1">Status</label>
          <select id="reg-status" class="w-full bg-slate-50 border border-slate-300 rounded px-2.5 py-1.5">
            <option value="PRESENT">PRESENT</option>
            <option value="WFH">WFH</option>
            <option value="ON_DUTY">ON_DUTY</option>
            <option value="ABSENT">ABSENT (Clears punch times)</option>
            <option value="LEAVE">LEAVE (Clears punch times)</option>
          </select>
        </div>
        <div>
          <label class="block font-semibold text-slate-600 mb-1">Punch-In (IST)</label>
          <input type="datetime-local" id="reg-punchin" class="w-full bg-slate-50 border border-slate-300 rounded px-2.5 py-1.5">
        </div>
        <div>
          <label class="block font-semibold text-slate-600 mb-1">Punch-Out (IST)</label>
          <input type="datetime-local" id="reg-punchout" class="w-full bg-slate-50 border border-slate-300 rounded px-2.5 py-1.5">
        </div>
        <div>
          <label class="block font-semibold text-slate-600 mb-1">Reason for Adjustment *</label>
          <input type="text" id="reg-reason" placeholder="e.g. Biometric device glitch" class="w-full bg-slate-50 border border-slate-300 rounded px-2.5 py-1.5">
        </div>
        <div>
          <label class="block font-semibold text-slate-600 mb-1">Regularized By *</label>
          <input type="text" id="reg-by" placeholder="e.g. hr.manager" class="w-full bg-slate-50 border border-slate-300 rounded px-2.5 py-1.5">
        </div>
      </div>

      <div id="reg-error" class="hidden text-xs text-rose-600 bg-rose-50 p-2 rounded"></div>

      <div class="flex justify-end space-x-2 pt-3 border-t border-slate-100">
        <button onclick="closeRegularizeModal()" class="px-3 py-1.5 rounded bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold">Cancel</button>
        <button onclick="submitRegularize()" class="px-4 py-1.5 rounded bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-semibold">Save Adjustment (PATCH)</button>
      </div>
    </div>
  </div>

  <!-- Modal: View History Audit Trail -->
  <div id="history-modal" class="hidden fixed inset-0 bg-slate-900/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
    <div class="bg-white rounded-2xl max-w-lg w-full p-6 shadow-2xl border border-slate-200 space-y-4">
      <div class="flex justify-between items-center pb-3 border-b border-slate-100">
        <h3 class="text-base font-bold text-slate-800 flex items-center">
          <i class="fa-solid fa-clock-rotate-left text-indigo-600 mr-2"></i> Audit Trail History
        </h3>
        <button onclick="closeHistoryModal()" class="text-slate-400 hover:text-slate-600 text-lg">&times;</button>
      </div>

      <div id="history-content" class="space-y-3 max-h-80 overflow-y-auto text-xs text-slate-700"></div>

      <div class="flex justify-end pt-3 border-t border-slate-100">
        <button onclick="closeHistoryModal()" class="px-4 py-1.5 rounded bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold">Close</button>
      </div>
    </div>
  </div>

  <!-- Modal: Add Employee -->
  <div id="add-emp-modal" class="hidden fixed inset-0 bg-slate-900/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
    <div class="bg-white rounded-2xl max-w-md w-full p-6 shadow-2xl border border-slate-200 space-y-4">
      <div class="flex justify-between items-center pb-3 border-b border-slate-100">
        <h3 class="text-base font-bold text-slate-800 flex items-center">
          <i class="fa-solid fa-user-plus text-indigo-600 mr-2"></i> Register New Employee
        </h3>
        <button onclick="closeAddEmpModal()" class="text-slate-400 hover:text-slate-600 text-lg">&times;</button>
      </div>

      <div class="space-y-3 text-xs">
        <div>
          <label class="block font-semibold text-slate-600 mb-1">Emp Code (e.g. EMP0010) *</label>
          <input type="text" id="new-emp-code" placeholder="EMP0010" class="w-full bg-slate-50 border border-slate-300 rounded px-2.5 py-1.5">
        </div>
        <div>
          <label class="block font-semibold text-slate-600 mb-1">Full Name *</label>
          <input type="text" id="new-emp-name" placeholder="Priya Sharma" class="w-full bg-slate-50 border border-slate-300 rounded px-2.5 py-1.5">
        </div>
        <div>
          <label class="block font-semibold text-slate-600 mb-1">Email *</label>
          <input type="email" id="new-emp-email" placeholder="priya@example.com" class="w-full bg-slate-50 border border-slate-300 rounded px-2.5 py-1.5">
        </div>
        <div>
          <label class="block font-semibold text-slate-600 mb-1">Department *</label>
          <input type="text" id="new-emp-dept" placeholder="Engineering" class="w-full bg-slate-50 border border-slate-300 rounded px-2.5 py-1.5">
        </div>
        <div class="grid grid-cols-2 gap-2">
          <div>
            <label class="block font-semibold text-slate-600 mb-1">Shift Start (HH:MM)</label>
            <input type="text" id="new-emp-start" value="09:30" class="w-full bg-slate-50 border border-slate-300 rounded px-2.5 py-1.5">
          </div>
          <div>
            <label class="block font-semibold text-slate-600 mb-1">Shift End (HH:MM)</label>
            <input type="text" id="new-emp-end" value="18:30" class="w-full bg-slate-50 border border-slate-300 rounded px-2.5 py-1.5">
          </div>
        </div>
        <div>
          <label class="block font-semibold text-slate-600 mb-1">Joined On (YYYY-MM-DD) *</label>
          <input type="date" id="new-emp-joined" value="2026-07-01" class="w-full bg-slate-50 border border-slate-300 rounded px-2.5 py-1.5">
        </div>
      </div>

      <div id="new-emp-error" class="hidden text-xs text-rose-600 bg-rose-50 p-2 rounded"></div>

      <div class="flex justify-end space-x-2 pt-3 border-t border-slate-100">
        <button onclick="closeAddEmpModal()" class="px-3 py-1.5 rounded bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold">Cancel</button>
        <button onclick="submitAddEmployee()" class="px-4 py-1.5 rounded bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-semibold">Create Employee</button>
      </div>
    </div>
  </div>

  <!-- JavaScript Frontend Controller -->
  <script>
    let employeesList = [];
    let currentAttPage = 1;
    let totalAttPages = 1;

    // Clock
    setInterval(() => {
      const now = new Date();
      const istString = now.toLocaleTimeString('en-US', { timeZone: 'Asia/Kolkata', hour12: false });
      document.getElementById('live-clock').innerText = istString + ' IST';
    }, 1000);

    // Health check
    async function checkHealth() {
      try {
        const res = await fetch('/health');
        if (res.ok) {
          document.getElementById('health-dot').className = 'w-2 h-2 rounded-full bg-emerald-400 mr-2';
          document.getElementById('health-text').innerText = 'MongoDB Connected (200)';
        } else {
          throw new Error();
        }
      } catch (e) {
        document.getElementById('health-dot').className = 'w-2 h-2 rounded-full bg-rose-400 mr-2';
        document.getElementById('health-text').innerText = 'Database Offline (503)';
      }
    }

    // Tabs
    function switchTab(tabId) {
      document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('bg-indigo-900', 'text-white', 'shadow-inner');
        btn.classList.add('text-indigo-200');
      });
      document.getElementById('tab-' + tabId).classList.add('active');
      const activeBtn = document.getElementById('tab-btn-' + tabId);
      activeBtn.classList.add('bg-indigo-900', 'text-white', 'shadow-inner');
      activeBtn.classList.remove('text-indigo-200');

      if (tabId === 'attendance') loadAttendanceLogs(currentAttPage);
      if (tabId === 'employees') loadEmployees();
    }

    function switchAnalyticsSubTab(subtabId) {
      document.querySelectorAll('.analytics-subtab').forEach(el => el.classList.add('hidden'));
      document.querySelectorAll('.subtab-btn').forEach(btn => {
        btn.classList.remove('bg-indigo-100', 'text-indigo-700');
        btn.classList.add('bg-slate-100', 'text-slate-600');
      });
      document.getElementById('subtab-' + subtabId).classList.remove('hidden');
      const activeBtn = document.getElementById('subtab-btn-' + subtabId);
      activeBtn.classList.add('bg-indigo-100', 'text-indigo-700');
      activeBtn.classList.remove('bg-slate-100', 'text-slate-600');
    }

    // Format timestamps
    function formatEpoch(ms) {
      if (!ms) return '-';
      const d = new Date(ms);
      return d.toLocaleTimeString('en-US', { timeZone: 'Asia/Kolkata', hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }) + ' IST';
    }

    // Load employees for select dropdowns & directory
    async function loadEmployees() {
      try {
        const res = await fetch('/employees?page=1&page_size=100');
        const data = await res.json();
        employeesList = data.items || [];

        // Populate dropdowns
        const inSelect = document.getElementById('punchin-emp');
        const outSelect = document.getElementById('punchout-emp');
        const mSelect = document.getElementById('monthly-emp');
        [inSelect, outSelect, mSelect].forEach(sel => {
          if (!sel) return;
          sel.innerHTML = employeesList.map(e => `<option value="${e.emp_code}">${e.emp_code} - ${e.name} (${e.department})</option>`).join('');
        });

        // Populate directory table
        const tbody = document.getElementById('employees-tbody');
        if (tbody) {
          tbody.innerHTML = employeesList.map(e => `
            <tr class="hover:bg-slate-50">
              <td class="px-4 py-3 font-mono font-bold text-indigo-700">${e.emp_code}</td>
              <td class="px-4 py-3 font-medium text-slate-800">${e.name}</td>
              <td class="px-4 py-3 text-slate-500">${e.email}</td>
              <td class="px-4 py-3 font-semibold text-slate-700">${e.department}</td>
              <td class="px-4 py-3 font-mono text-slate-600">${e.shift_start} - ${e.shift_end}</td>
              <td class="px-4 py-3 text-slate-500">${e.joined_on}</td>
            </tr>
          `).join('');
        }
      } catch (e) {
        console.error(e);
      }
    }

    // Punch In (Uses current instant automatically)
    async function submitPunchIn() {
      const emp_code = document.getElementById('punchin-emp').value;
      const status = document.getElementById('punchin-status').value;
      const resBox = document.getElementById('punch-result');

      if (!emp_code) {
        alert('Please select an employee first.');
        return;
      }

      // Automatically capture current time
      const body = {
        emp_code: emp_code,
        status: status,
        punched_at: Date.now()
      };

      try {
        const res = await fetch('/attendance/punch-in', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        });
        const data = await res.json();
        resBox.classList.remove('hidden', 'bg-emerald-50', 'border-emerald-200', 'text-emerald-800', 'bg-rose-50', 'border-rose-200', 'text-rose-800');
        if (res.ok) {
          resBox.classList.add('bg-emerald-50', 'border-emerald-200', 'text-emerald-800');
          resBox.innerHTML = `<strong>Punch-In Success (201):</strong> Recorded punch for <b>${data.emp_code}</b> on <b>${data.date}</b> at <b>${formatEpoch(data.punch_in)}</b>. Status: <b>${data.status}</b>, Late Minutes: <b>${data.late_minutes} min</b>.`;
        } else {
          resBox.classList.add('bg-rose-50', 'border-rose-200', 'text-rose-800');
          resBox.innerHTML = `<strong>Punch-In Failed (${res.status}):</strong> ${data.detail || JSON.stringify(data)}`;
        }
      } catch (err) {
        alert('Network request failed');
      }
    }

    // Punch Out (Uses current instant automatically)
    async function submitPunchOut() {
      const emp_code = document.getElementById('punchout-emp').value;
      const resBox = document.getElementById('punch-result');

      if (!emp_code) {
        alert('Please select an employee first.');
        return;
      }

      // Automatically capture current time
      const body = {
        emp_code: emp_code,
        punched_at: Date.now()
      };

      try {
        const res = await fetch('/attendance/punch-out', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        });
        const data = await res.json();
        resBox.classList.remove('hidden', 'bg-emerald-50', 'border-emerald-200', 'text-emerald-800', 'bg-rose-50', 'border-rose-200', 'text-rose-800');
        if (res.ok) {
          resBox.classList.add('bg-emerald-50', 'border-emerald-200', 'text-emerald-800');
          resBox.innerHTML = `<strong>Punch-Out Success (200):</strong> Closed shift for <b>${data.emp_code}</b> on <b>${data.date}</b>. Work Hours: <b>${data.work_hours} hrs</b>, Overtime: <b>${data.overtime_minutes} mins</b>, Half Day: <b>${data.half_day}</b>.`;
        } else {
          resBox.classList.add('bg-rose-50', 'border-rose-200', 'text-rose-800');
          resBox.innerHTML = `<strong>Error (${res.status}):</strong> ${data.detail || JSON.stringify(data)}`;
        }
      } catch (err) {
        alert('Network request failed');
      }
    }

    // Load Attendance Logs
    async function loadAttendanceLogs(page) {
      currentAttPage = page;
      const emp = document.getElementById('att-filter-emp').value.trim();
      const from = document.getElementById('att-filter-from').value;
      const to = document.getElementById('att-filter-to').value;
      const status = document.getElementById('att-filter-status').value;

      let url = `/attendance?page=${page}&page_size=15`;
      if (emp) url += `&emp_code=${emp}`;
      if (from) url += `&date_from=${from}`;
      if (to) url += `&date_to=${to}`;
      if (status) url += `&status=${status}`;

      try {
        const res = await fetch(url);
        const data = await res.json();
        const tbody = document.getElementById('attendance-tbody');

        if (!data.items || data.items.length === 0) {
          tbody.innerHTML = '<tr><td colspan="10" class="text-center py-6 text-slate-400">No attendance records found matching filters.</td></tr>';
          document.getElementById('att-pagination-info').innerText = 'No records';
          document.getElementById('att-prev-btn').disabled = true;
          document.getElementById('att-next-btn').disabled = true;
          return;
        }

        totalAttPages = Math.ceil(data.total / data.page_size);
        document.getElementById('att-pagination-info').innerText = `Page ${data.page} of ${totalAttPages} (${data.total} total records)`;
        document.getElementById('att-prev-btn').disabled = (data.page <= 1);
        document.getElementById('att-next-btn').disabled = (data.page >= totalAttPages);

        tbody.innerHTML = data.items.map(item => {
          const statusColors = {
            'PRESENT': 'bg-emerald-100 text-emerald-800',
            'WFH': 'bg-blue-100 text-blue-800',
            'ON_DUTY': 'bg-purple-100 text-purple-800',
            'ABSENT': 'bg-rose-100 text-rose-800',
            'LEAVE': 'bg-amber-100 text-amber-800'
          };
          const badgeClass = statusColors[item.status] || 'bg-slate-100 text-slate-800';
          const hasHistory = item.history && item.history.length > 0;

          return `
            <tr class="hover:bg-slate-50">
              <td class="px-4 py-3 font-mono font-medium text-slate-700">${item.date}</td>
              <td class="px-4 py-3 font-mono font-bold text-indigo-700">${item.emp_code}</td>
              <td class="px-4 py-3"><span class="px-2 py-0.5 rounded text-xs font-semibold ${badgeClass}">${item.status}</span></td>
              <td class="px-4 py-3 font-mono text-slate-600">${formatEpoch(item.punch_in)}</td>
              <td class="px-4 py-3 font-mono text-slate-600">${formatEpoch(item.punch_out)}</td>
              <td class="px-4 py-3 font-semibold text-slate-800">${item.work_hours !== null ? item.work_hours + ' h' : '-'}</td>
              <td class="px-4 py-3 ${item.late_minutes > 0 ? 'font-bold text-rose-600' : 'text-slate-400'}">${item.late_minutes}</td>
              <td class="px-4 py-3 ${item.overtime_minutes > 0 ? 'font-bold text-emerald-600' : 'text-slate-400'}">${item.overtime_minutes}</td>
              <td class="px-4 py-3">${item.half_day ? '<span class="px-1.5 py-0.5 rounded bg-amber-100 text-amber-800 font-bold text-[10px]">HALF DAY</span>' : '-'}</td>
              <td class="px-4 py-3 text-right space-x-1 whitespace-nowrap">
                <button onclick="openRegularizeModal('${item.emp_code}', '${item.date}', '${item.status}', ${item.punch_in}, ${item.punch_out})" class="text-xs text-indigo-600 hover:text-indigo-800 font-semibold px-2 py-1 rounded bg-indigo-50 border border-indigo-200">
                  <i class="fa-solid fa-pen-to-square"></i> Regularize
                </button>
                ${hasHistory ? `
                <button onclick='openHistoryModal(${JSON.stringify(item.history).replace(/'/g, "&apos;")})' class="text-xs text-slate-600 hover:text-slate-800 px-2 py-1 rounded bg-slate-100 border border-slate-300">
                  <i class="fa-solid fa-clock-rotate-left"></i> History (${item.history.length})
                </button>` : ''}
              </td>
            </tr>
          `;
        }).join('');
      } catch (e) {
        console.error(e);
      }
    }

    function prevAttPage() { if (currentAttPage > 1) loadAttendanceLogs(currentAttPage - 1); }
    function nextAttPage() { if (currentAttPage < totalAttPages) loadAttendanceLogs(currentAttPage + 1); }

    // Regularize Modal
    let activeRegEmp = '';
    let activeRegDate = '';

    function openRegularizeModal(emp, dateVal, currentStatus, inMs, outMs) {
      activeRegEmp = emp;
      activeRegDate = dateVal;
      document.getElementById('reg-emp').innerText = emp;
      document.getElementById('reg-date').innerText = dateVal;
      document.getElementById('reg-status').value = currentStatus || 'PRESENT';

      if (inMs) {
        const d = new Date(inMs);
        document.getElementById('reg-punchin').value = d.toISOString().slice(0, 16);
      } else {
        document.getElementById('reg-punchin').value = '';
      }

      if (outMs) {
        const d = new Date(outMs);
        document.getElementById('reg-punchout').value = d.toISOString().slice(0, 16);
      } else {
        document.getElementById('reg-punchout').value = '';
      }

      document.getElementById('reg-reason').value = '';
      document.getElementById('reg-by').value = 'hr.admin';
      document.getElementById('reg-error').classList.add('hidden');
      document.getElementById('regularize-modal').classList.remove('hidden');
    }

    function closeRegularizeModal() {
      document.getElementById('regularize-modal').classList.add('hidden');
    }

    async function submitRegularize() {
      const status = document.getElementById('reg-status').value;
      const inVal = document.getElementById('reg-punchin').value;
      const outVal = document.getElementById('reg-punchout').value;
      const reason = document.getElementById('reg-reason').value.trim();
      const regularized_by = document.getElementById('reg-by').value.trim();
      const errBox = document.getElementById('reg-error');

      const body = { reason, regularized_by, status };
      if (inVal && status !== 'ABSENT' && status !== 'LEAVE') body.punch_in = new Date(inVal).getTime();
      if (outVal && status !== 'ABSENT' && status !== 'LEAVE') body.punch_out = new Date(outVal).getTime();

      try {
        const res = await fetch(`/attendance/${activeRegEmp}/${activeRegDate}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        });
        const data = await res.json();
        if (res.ok) {
          closeRegularizeModal();
          loadAttendanceLogs(currentAttPage);
        } else {
          errBox.classList.remove('hidden');
          errBox.innerText = data.detail || 'Regularization failed';
        }
      } catch (e) {
        errBox.classList.remove('hidden');
        errBox.innerText = 'Network request failed';
      }
    }

    // History Modal
    function openHistoryModal(historyList) {
      const container = document.getElementById('history-content');
      if (!historyList || historyList.length === 0) {
        container.innerHTML = '<p class="text-slate-400">No audit history recorded.</p>';
      } else {
        container.innerHTML = historyList.map(h => `
          <div class="p-3 bg-slate-50 border border-slate-200 rounded-lg space-y-1">
            <div class="flex justify-between font-semibold text-slate-800">
              <span>By: ${h.by}</span>
              <span class="text-slate-500 font-mono">${new Date(h.at).toLocaleDateString()} ${formatEpoch(h.at)}</span>
            </div>
            <p class="text-slate-600"><strong>Reason:</strong> ${h.reason}</p>
            <div class="bg-white p-2 rounded border border-slate-100 mt-2 space-y-1 font-mono text-[11px]">
              ${Object.entries(h.changes).map(([k, v]) => `
                <div><span class="text-indigo-600 font-semibold">${k}:</span> <span class="text-rose-500">${v.from !== null ? (k.includes('punch') ? formatEpoch(v.from) : v.from) : 'null'}</span> &rarr; <span class="text-emerald-600 font-bold">${v.to !== null ? (k.includes('punch') ? formatEpoch(v.to) : v.to) : 'null'}</span></div>
              `).join('')}
            </div>
          </div>
        `).join('');
      }
      document.getElementById('history-modal').classList.remove('hidden');
    }

    function closeHistoryModal() {
      document.getElementById('history-modal').classList.add('hidden');
    }

    // Add Employee Modal
    function openAddEmpModal() {
      document.getElementById('new-emp-error').classList.add('hidden');
      document.getElementById('add-emp-modal').classList.remove('hidden');
    }
    function closeAddEmpModal() {
      document.getElementById('add-emp-modal').classList.add('hidden');
    }
    async function submitAddEmployee() {
      const emp_code = document.getElementById('new-emp-code').value.trim();
      const name = document.getElementById('new-emp-name').value.trim();
      const email = document.getElementById('new-emp-email').value.trim();
      const department = document.getElementById('new-emp-dept').value.trim();
      const shift_start = document.getElementById('new-emp-start').value.trim();
      const shift_end = document.getElementById('new-emp-end').value.trim();
      const joined_on = document.getElementById('new-emp-joined').value;
      const errBox = document.getElementById('new-emp-error');

      try {
        const res = await fetch('/employees', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ emp_code, name, email, department, shift_start, shift_end, joined_on })
        });
        const data = await res.json();
        if (res.ok) {
          closeAddEmpModal();
          loadEmployees();
        } else {
          errBox.classList.remove('hidden');
          errBox.innerText = data.detail || 'Creation failed';
        }
      } catch (e) {
        errBox.classList.remove('hidden');
        errBox.innerText = 'Network error';
      }
    }

    // Analytics: Department Summary
    async function loadDeptSummary() {
      const month = document.getElementById('dept-summary-month').value;
      const tbody = document.getElementById('dept-summary-tbody');
      tbody.innerHTML = '<tr><td colspan="8" class="text-center py-4 text-slate-400">Loading department summary...</td></tr>';

      try {
        const res = await fetch(`/analytics/departments/summary?month=${month}`);
        const data = await res.json();
        if (!data.items || data.items.length === 0) {
          tbody.innerHTML = '<tr><td colspan="8" class="text-center py-4 text-slate-400">No departments found for this month.</td></tr>';
          return;
        }
        tbody.innerHTML = data.items.map(d => `
          <tr class="hover:bg-slate-50">
            <td class="px-4 py-3 font-semibold text-slate-800">${d.department}</td>
            <td class="px-4 py-3 font-mono text-indigo-700 font-bold">${d.headcount}</td>
            <td class="px-4 py-3 text-emerald-700 font-bold">${d.present_days}</td>
            <td class="px-4 py-3">${d.avg_work_hours !== null ? d.avg_work_hours + ' h' : '-'}</td>
            <td class="px-4 py-3 font-semibold text-rose-600">${d.late_count}</td>
            <td class="px-4 py-3 font-mono">${d.total_late_minutes}</td>
            <td class="px-4 py-3 font-mono">${d.leave_count}</td>
            <td class="px-4 py-3 font-mono">${d.on_duty_count}</td>
          </tr>
        `).join('');
      } catch (e) {
        tbody.innerHTML = '<tr><td colspan="8" class="text-center py-4 text-rose-500">Failed to load department summary</td></tr>';
      }
    }

    // Analytics: Late Leaderboard
    async function loadLeaderboard() {
      const month = document.getElementById('leaderboard-month').value;
      const limit = document.getElementById('leaderboard-limit').value || 10;
      const tbody = document.getElementById('leaderboard-tbody');
      tbody.innerHTML = '<tr><td colspan="6" class="text-center py-4 text-slate-400">Calculating late leaderboard...</td></tr>';

      try {
        const res = await fetch(`/analytics/leaderboard/late?month=${month}&limit=${limit}`);
        const data = await res.json();
        if (!data.items || data.items.length === 0) {
          tbody.innerHTML = '<tr><td colspan="6" class="text-center py-4 text-slate-400">No late records in this month!</td></tr>';
          return;
        }
        tbody.innerHTML = data.items.map(item => {
          let rankBadge = `<span class="px-2 py-0.5 rounded font-bold text-xs bg-slate-100 text-slate-700">#${item.rank}</span>`;
          if (item.rank === 1) rankBadge = `<span class="px-2 py-0.5 rounded font-bold text-xs bg-amber-100 text-amber-800 border border-amber-300"><i class="fa-solid fa-trophy mr-1 text-amber-600"></i> Rank 1</span>`;
          if (item.rank === 2) rankBadge = `<span class="px-2 py-0.5 rounded font-bold text-xs bg-slate-200 text-slate-800 border border-slate-300">Rank 2</span>`;
          if (item.rank === 3) rankBadge = `<span class="px-2 py-0.5 rounded font-bold text-xs bg-orange-100 text-orange-800 border border-orange-300">Rank 3</span>`;

          return `
            <tr class="hover:bg-slate-50">
              <td class="px-4 py-3">${rankBadge}</td>
              <td class="px-4 py-3 font-mono font-bold text-indigo-700">${item.emp_code}</td>
              <td class="px-4 py-3 font-medium text-slate-800">${item.name}</td>
              <td class="px-4 py-3 text-slate-600">${item.department}</td>
              <td class="px-4 py-3 font-bold text-rose-600 font-mono">${item.total_late_minutes} mins</td>
              <td class="px-4 py-3 font-semibold text-slate-700">${item.late_count} times</td>
            </tr>
          `;
        }).join('');
      } catch (e) {
        tbody.innerHTML = '<tr><td colspan="6" class="text-center py-4 text-rose-500">Failed to load leaderboard</td></tr>';
      }
    }

    // Analytics: Department Trend
    async function loadTrend() {
      const dept = document.getElementById('trend-dept').value;
      const from = document.getElementById('trend-from').value;
      const to = document.getElementById('trend-to').value;
      const tbody = document.getElementById('trend-tbody');
      tbody.innerHTML = '<tr><td colspan="7" class="text-center py-4 text-slate-400">Running $densify gap-filling aggregation...</td></tr>';

      try {
        const res = await fetch(`/analytics/departments/${dept}/trend?from=${from}&to=${to}`);
        const data = await res.json();
        if (!data.items || data.items.length === 0) {
          tbody.innerHTML = '<tr><td colspan="7" class="text-center py-4 text-slate-400">No trend data found.</td></tr>';
          return;
        }
        tbody.innerHTML = data.items.map(t => `
          <tr class="hover:bg-slate-50 ${!t.is_working_day ? 'bg-slate-50/50' : ''}">
            <td class="px-4 py-3 font-mono font-medium">${t.date}</td>
            <td class="px-4 py-3">${t.is_working_day ? '<span class="text-emerald-700 font-semibold">Weekday</span>' : '<span class="text-slate-400">Weekend</span>'}</td>
            <td class="px-4 py-3 font-mono">${t.headcount}</td>
            <td class="px-4 py-3 font-semibold ${t.present_count > 0 ? 'text-indigo-600' : 'text-slate-400'}">${t.present_count}</td>
            <td class="px-4 py-3 font-semibold ${t.late_count > 0 ? 'text-rose-600' : 'text-slate-400'}">${t.late_count}</td>
            <td class="px-4 py-3 font-mono">${t.attendance_rate !== null ? (t.attendance_rate * 100).toFixed(1) + '%' : '-'}</td>
            <td class="px-4 py-3 font-mono font-bold text-indigo-700">${t.moving_avg_7d !== null ? (t.moving_avg_7d * 100).toFixed(1) + '%' : '-'}</td>
          </tr>
        `).join('');
      } catch (e) {
        tbody.innerHTML = '<tr><td colspan="7" class="text-center py-4 text-rose-500">Failed to load trend</td></tr>';
      }
    }

    // Analytics: Employee Monthly
    async function loadEmpMonthly() {
      const emp = document.getElementById('monthly-emp').value;
      const month = document.getElementById('monthly-month').value;
      const cards = document.getElementById('emp-monthly-cards');

      try {
        const res = await fetch(`/analytics/employees/${emp}/monthly?month=${month}`);
        const data = await res.json();
        cards.classList.remove('hidden');
        document.getElementById('m-card-working').innerText = data.working_days;
        document.getElementById('m-card-present').innerText = data.present_days;
        document.getElementById('m-card-pct').innerText = data.attendance_pct !== null ? data.attendance_pct + '%' : 'N/A';
        document.getElementById('m-card-late').innerText = data.total_late_minutes + ' m (' + data.late_count + 'x)';
      } catch (e) {
        alert('Failed to load employee monthly summary');
      }
    }

    // Initialize
    window.addEventListener('DOMContentLoaded', () => {
      checkHealth();
      loadEmployees();
    });
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def get_frontend():
    """Serves the interactive web UI dashboard."""
    return HTMLResponse(content=FRONTEND_HTML)

