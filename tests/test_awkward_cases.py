import concurrent.futures
from datetime import datetime, timezone, timedelta
import pytest
from fastapi.testclient import TestClient

from app.main import app, db, init_indexes

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_db():
    init_indexes()


def test_overnight_shift_and_date_r1():
    code = "EMP8001"
    db.employees.delete_one({"emp_code": code})
    db.attendance_logs.delete_many({"emp_code": code})

    # Overnight shift: 22:00 to 06:00
    client.post("/employees", json={
        "emp_code": code,
        "name": "Night Owl",
        "email": "night@example.com",
        "department": "Security",
        "shift_start": "22:00",
        "shift_end": "06:00",
        "joined_on": "2026-01-01"
    })

    # Punch in at 01:15 IST on 2026-08-11
    # Since 01:15 < 06:00, date belongs to yesterday: 2026-08-10!
    # 2026-08-11 01:15:00 IST = 2026-08-10 19:45:00 UTC
    p_in_ms = int(datetime(2026, 8, 10, 19, 45, 0, tzinfo=timezone.utc).timestamp() * 1000)
    res_in = client.post("/attendance/punch-in", json={
        "emp_code": code,
        "punched_at": p_in_ms,
        "status": "PRESENT"
    })
    assert res_in.status_code == 201
    assert res_in.json()["date"] == "2026-08-10"

    # Punch out at 06:50 IST on 2026-08-11 (50 min overtime past 06:00)
    # 2026-08-11 06:50:00 IST = 2026-08-11 01:20:00 UTC
    p_out_ms = int(datetime(2026, 8, 11, 1, 20, 0, tzinfo=timezone.utc).timestamp() * 1000)
    res_out = client.post("/attendance/punch-out", json={
        "emp_code": code,
        "punched_at": p_out_ms
    })
    assert res_out.status_code == 200
    data = res_out.json()
    assert data["overtime_minutes"] == 50
    # Work hours: 01:15 to 06:50 is 5h 35m = 5.58 hours
    assert data["work_hours"] == 5.58


def test_leaderboard_ties_and_rank_skipping():
    db.employees.delete_many({"emp_code": {"$in": ["EMP8101", "EMP8102", "EMP8103"]}})
    db.attendance_logs.delete_many({"emp_code": {"$in": ["EMP8101", "EMP8102", "EMP8103"]}})

    for c, name in [("EMP8101", "Alice"), ("EMP8102", "Bob"), ("EMP8103", "Charlie")]:
        client.post("/employees", json={
            "emp_code": c,
            "name": name,
            "email": f"{c.lower()}@example.com",
            "department": "Finance",
            "shift_start": "09:30",
            "shift_end": "18:30",
            "joined_on": "2026-01-01"
        })

    # Alice and Bob both have 45 late minutes on 2026-09-01
    # 10:15:00 IST = 04:45:00 UTC
    p_late_45 = int(datetime(2026, 9, 1, 4, 45, 0, tzinfo=timezone.utc).timestamp() * 1000)
    client.post("/attendance/punch-in", json={"emp_code": "EMP8101", "punched_at": p_late_45})
    client.post("/attendance/punch-in", json={"emp_code": "EMP8102", "punched_at": p_late_45})

    # Charlie has 20 late minutes
    # 09:50:00 IST = 04:20:00 UTC
    p_late_20 = int(datetime(2026, 9, 1, 4, 20, 0, tzinfo=timezone.utc).timestamp() * 1000)
    client.post("/attendance/punch-in", json={"emp_code": "EMP8103", "punched_at": p_late_20})

    # Check leaderboard:
    # Alice and Bob tie at Rank 1. Next rank Charlie must be Rank 3!
    res = client.get("/analytics/leaderboard/late?month=2026-09&department=Finance")
    assert res.status_code == 200
    items = res.json()["items"]
    assert len(items) == 3
    assert items[0]["rank"] == 1
    assert items[1]["rank"] == 1
    assert items[0]["emp_code"] == "EMP8101"
    assert items[1]["emp_code"] == "EMP8102"
    assert items[2]["rank"] == 3
    assert items[2]["emp_code"] == "EMP8103"

    # Test limit=1: ties at cutoff must BOTH be returned!
    res_lim1 = client.get("/analytics/leaderboard/late?month=2026-09&department=Finance&limit=1")
    assert res_lim1.status_code == 200
    items_lim1 = res_lim1.json()["items"]
    assert len(items_lim1) == 2  # Both rank 1 are returned
    assert items_lim1[0]["rank"] == 1
    assert items_lim1[1]["rank"] == 1


def test_department_summary_zero_logs_and_midmonth_joiner():
    dept = "Logistics"
    db.employees.delete_many({"department": dept})
    db.attendance_logs.delete_many({"emp_code": {"$in": ["EMP8201", "EMP8202", "EMP8203"]}})

    # EMP8201: joined 2026-01-01 (0 logs in October)
    client.post("/employees", json={
        "emp_code": "EMP8201",
        "name": "Zero Logs",
        "email": "zero@example.com",
        "department": dept,
        "shift_start": "09:30",
        "shift_end": "18:30",
        "joined_on": "2026-01-01"
    })
    # EMP8202: joined 2026-10-15 (mid-month joiner)
    client.post("/employees", json={
        "emp_code": "EMP8202",
        "name": "Mid Month",
        "email": "mid@example.com",
        "department": dept,
        "shift_start": "09:30",
        "shift_end": "18:30",
        "joined_on": "2026-10-15"
    })
    # EMP8203: joins 2026-11-01 (future joiner, must be excluded from October)
    client.post("/employees", json={
        "emp_code": "EMP8203",
        "name": "Future Joiner",
        "email": "future@example.com",
        "department": dept,
        "shift_start": "09:30",
        "shift_end": "18:30",
        "joined_on": "2026-11-01"
    })

    # Query October 2026 summary:
    # Headcount must be 2 (EMP8201 and EMP8202), excluding EMP8203!
    res = client.get(f"/analytics/departments/summary?month=2026-10&department={dept}")
    assert res.status_code == 200
    items = res.json()["items"]
    assert len(items) == 1
    assert items[0]["department"] == dept
    assert items[0]["headcount"] == 2
    assert items[0]["present_days"] == 0.0
    assert items[0]["avg_work_hours"] is None


def test_concurrent_punch_in():
    code = "EMP8301"
    db.employees.delete_one({"emp_code": code})
    db.attendance_logs.delete_many({"emp_code": code})

    client.post("/employees", json={
        "emp_code": code,
        "name": "Race Tester",
        "email": "race@example.com",
        "department": "R&D",
        "shift_start": "09:30",
        "shift_end": "18:30",
        "joined_on": "2026-01-01"
    })

    p_ms = int(datetime(2026, 10, 5, 4, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)

    def do_punch():
        # Fresh client per thread
        c = TestClient(app)
        return c.post("/attendance/punch-in", json={
            "emp_code": code,
            "punched_at": p_ms,
            "status": "PRESENT"
        }).status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(lambda _: do_punch(), range(5)))

    # Exactly one 201, rest 409
    assert results.count(201) == 1
    assert results.count(409) == 4


def test_concurrent_punch_out():
    code = "EMP8302"
    db.employees.delete_one({"emp_code": code})
    db.attendance_logs.delete_many({"emp_code": code})

    client.post("/employees", json={
        "emp_code": code,
        "name": "Race Tester 2",
        "email": "race2@example.com",
        "department": "R&D",
        "shift_start": "09:30",
        "shift_end": "18:30",
        "joined_on": "2026-01-01"
    })

    p_in_ms = int(datetime(2026, 10, 5, 4, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
    p_out_ms = int(datetime(2026, 10, 5, 13, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)

    client.post("/attendance/punch-in", json={"emp_code": code, "punched_at": p_in_ms})

    def do_punch_out():
        c = TestClient(app)
        return c.post("/attendance/punch-out", json={
            "emp_code": code,
            "punched_at": p_out_ms
        }).status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(lambda _: do_punch_out(), range(5)))

    # Exactly one 200, rest 409
    assert results.count(200) == 1
    assert results.count(409) == 4
