import calendar
from datetime import datetime, timezone, timedelta
import pytest
from fastapi.testclient import TestClient

from app.main import app, db, init_indexes

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_db():
    init_indexes()


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_create_and_list_employee():
    code = "EMP9999"
    db.employees.delete_one({"emp_code": code})

    # Test invalid: equal shifts
    res = client.post("/employees", json={
        "emp_code": code,
        "name": "Test User",
        "email": "test@example.com",
        "department": "QA",
        "shift_start": "09:30",
        "shift_end": "09:30",
        "joined_on": "2026-01-01"
    })
    assert res.status_code == 422

    # Test valid create
    res = client.post("/employees", json={
        "emp_code": code,
        "name": "Test User",
        "email": "test@example.com",
        "department": "QA",
        "shift_start": "09:30",
        "shift_end": "18:30",
        "joined_on": "2026-01-01"
    })
    assert res.status_code == 201
    data = res.json()
    assert data["emp_code"] == code
    assert isinstance(data["created_at"], int)
    assert "_id" not in data

    # Test duplicate 409
    res_dup = client.post("/employees", json={
        "emp_code": code,
        "name": "Duplicate User",
        "email": "dup@example.com",
        "department": "QA",
        "shift_start": "09:30",
        "shift_end": "18:30",
        "joined_on": "2026-01-01"
    })
    assert res_dup.status_code == 409

    # Test list
    res_list = client.get("/employees?department=QA")
    assert res_list.status_code == 200
    list_data = res_list.json()
    assert list_data["total"] >= 1
    assert any(e["emp_code"] == code for e in list_data["items"])


def test_punch_in_and_out_and_rules():
    code = "EMP9998"
    db.employees.delete_one({"emp_code": code})
    db.attendance_logs.delete_many({"emp_code": code})

    # Create employee with 09:30 - 18:30
    client.post("/employees", json={
        "emp_code": code,
        "name": "Punch Tester",
        "email": "punch@example.com",
        "department": "Testing",
        "shift_start": "09:30",
        "shift_end": "18:30",
        "joined_on": "2026-07-01"
    })

    # 1. Punch in at exactly 09:40:00 (10 min -> grace, late_minutes=0)
    # 2026-07-20 09:40:00 IST = 2026-07-20 04:10:00 UTC
    p_in_exact_grace = int(datetime(2026, 7, 20, 4, 10, 0, tzinfo=timezone.utc).timestamp() * 1000)
    res_in = client.post("/attendance/punch-in", json={
        "emp_code": code,
        "punched_at": p_in_exact_grace,
        "status": "PRESENT"
    })
    assert res_in.status_code == 201
    data_in = res_in.json()
    assert data_in["date"] == "2026-07-20"
    assert data_in["late_minutes"] == 0
    assert data_in["punch_out"] is None

    # Duplicate punch in on same day -> 409
    res_in_dup = client.post("/attendance/punch-in", json={
        "emp_code": code,
        "punched_at": p_in_exact_grace + 10000,
        "status": "PRESENT"
    })
    assert res_in_dup.status_code == 409

    # 2. Punch out: 19:10:00 IST (40 min after shift end 18:30 -> overtime=40)
    # 2026-07-20 19:10:00 IST = 2026-07-20 13:40:00 UTC
    p_out = int(datetime(2026, 7, 20, 13, 40, 0, tzinfo=timezone.utc).timestamp() * 1000)
    res_out = client.post("/attendance/punch-out", json={
        "emp_code": code,
        "punched_at": p_out
    })
    assert res_out.status_code == 200
    data_out = res_out.json()
    assert data_out["overtime_minutes"] == 40
    # Work hours: 09:40 to 19:10 is 9.5 hours
    assert data_out["work_hours"] == 9.5
    assert data_out["half_day"] is False

    # Second punch out -> 409
    res_out_dup = client.post("/attendance/punch-out", json={
        "emp_code": code,
        "punched_at": p_out + 1000
    })
    assert res_out_dup.status_code == 409


def test_punch_in_strictly_late_rule_r2():
    code = "EMP9997"
    db.employees.delete_one({"emp_code": code})
    db.attendance_logs.delete_many({"emp_code": code})

    client.post("/employees", json={
        "emp_code": code,
        "name": "Late Tester",
        "email": "late@example.com",
        "department": "Testing",
        "shift_start": "09:30",
        "shift_end": "18:30",
        "joined_on": "2026-07-01"
    })

    # Punch in at 09:40:01 IST (> 10:00 grace -> late = floor(10.016) = 10)
    # 2026-07-21 09:40:01 IST = 2026-07-21 04:10:01 UTC
    p_in_late = int(datetime(2026, 7, 21, 4, 10, 1, tzinfo=timezone.utc).timestamp() * 1000)
    res_in = client.post("/attendance/punch-in", json={
        "emp_code": code,
        "punched_at": p_in_late,
        "status": "PRESENT"
    })
    assert res_in.status_code == 201
    assert res_in.json()["late_minutes"] == 10


def test_regularize_patch():
    code = "EMP9996"
    db.employees.delete_one({"emp_code": code})
    db.attendance_logs.delete_many({"emp_code": code})

    client.post("/employees", json={
        "emp_code": code,
        "name": "Reg Tester",
        "email": "reg@example.com",
        "department": "Testing",
        "shift_start": "09:30",
        "shift_end": "18:30",
        "joined_on": "2026-07-01"
    })

    # Punch in 09:28 IST (03:58 UTC), punch out 18:35 IST (13:05 UTC)
    p_in = int(datetime(2026, 7, 6, 3, 58, 0, tzinfo=timezone.utc).timestamp() * 1000)
    p_out = int(datetime(2026, 7, 6, 13, 5, 0, tzinfo=timezone.utc).timestamp() * 1000)
    client.post("/attendance/punch-in", json={"emp_code": code, "punched_at": p_in})
    client.post("/attendance/punch-out", json={"emp_code": code, "punched_at": p_out})

    # Patch reason and change punch_out to 19:15 IST (13:45 UTC)
    new_out_ms = int(datetime(2026, 7, 6, 13, 45, 0, tzinfo=timezone.utc).timestamp() * 1000)
    res = client.patch(f"/attendance/{code}/2026-07-06", json={
        "punch_out": new_out_ms,
        "reason": "Manager approved late work",
        "regularized_by": "manager.bob"
    })
    assert res.status_code == 200
    data = res.json()
    assert len(data["history"]) >= 1
    last_hist = data["history"][-1]
    assert last_hist["by"] == "manager.bob"
    assert "punch_out" in last_hist["changes"]
    assert "work_hours" in last_hist["changes"]
    assert "overtime_minutes" in last_hist["changes"]
    assert data["overtime_minutes"] == 45


def test_analytics_endpoints():
    # 1. Employee monthly
    res_emp = client.get("/analytics/employees/EMP0001/monthly?month=2026-07")
    assert res_emp.status_code == 200
    data_emp = res_emp.json()
    assert data_emp["emp_code"] == "EMP0001"
    assert data_emp["working_days"] == 23  # July 2026 Mon-Fri count
    assert data_emp["present_days"] >= 1.0

    # 2. Department summary
    res_dept = client.get("/analytics/departments/summary?month=2026-07")
    assert res_dept.status_code == 200
    data_dept = res_dept.json()
    assert "items" in data_dept
    depts = [d["department"] for d in data_dept["items"]]
    assert "Engineering" in depts
    assert "Sales" in depts

    # 3. Late leaderboard
    res_lb = client.get("/analytics/leaderboard/late?month=2026-07&limit=5")
    assert res_lb.status_code == 200
    data_lb = res_lb.json()
    assert "items" in data_lb
    assert len(data_lb["items"]) >= 1
    assert data_lb["items"][0]["rank"] == 1
    assert data_lb["items"][0]["total_late_minutes"] > 0

    # 4. Department trend
    res_tr = client.get("/analytics/departments/Engineering/trend?from=2026-07-06&to=2026-07-10")
    assert res_tr.status_code == 200
    data_tr = res_tr.json()
    assert data_tr["department"] == "Engineering"
    assert len(data_tr["items"]) == 5
    for item in data_tr["items"]:
        assert "date" in item
        assert "headcount" in item
        assert "is_working_day" in item
        assert "attendance_rate" in item
        assert "moving_avg_7d" in item


def test_admin_explain():
    endpoints = [
        "/admin/explain/attendance_list?page=1&page_size=10",
        "/admin/explain/employee_monthly?emp_code=EMP0001&month=2026-07",
        "/admin/explain/department_summary?month=2026-07",
        "/admin/explain/late_leaderboard?month=2026-07&limit=10",
        "/admin/explain/department_trend?department=Engineering&from=2026-07-01&to=2026-07-10",
    ]
    for url in endpoints:
        res = client.get(url)
        assert res.status_code == 200, f"Failed explain for {url}: {res.text}"
        data = res.json()
        assert "endpoint" in data
        assert "collection" in data
        assert "explain" in data
