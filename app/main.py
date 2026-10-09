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
