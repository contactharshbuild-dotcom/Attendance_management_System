import json
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_explain_all_endpoints():
    test_cases = [
        (
            "attendance_list",
            "/admin/explain/attendance_list?emp_code=EMP0001&date_from=2026-07-01&date_to=2026-07-31&page=1&page_size=20",
            "attendance_logs"
        ),
        (
            "employee_monthly",
            "/admin/explain/employee_monthly?emp_code=EMP0001&month=2026-07",
            "attendance_logs"
        ),
        (
            "department_summary",
            "/admin/explain/department_summary?month=2026-07&department=Engineering",
            "employees"
        ),
        (
            "late_leaderboard",
            "/admin/explain/late_leaderboard?month=2026-07&limit=10",
            "attendance_logs"
        ),
        (
            "department_trend",
            "/admin/explain/department_trend?department=Engineering&from=2026-07-01&to=2026-07-15",
            "attendance_logs"
        ),
    ]

    for name, url, expected_coll in test_cases:
        res = client.get(url)
        assert res.status_code == 200, f"Explain failed for {name}: {res.text}"
        data = res.json()
        assert data["endpoint"] == name
        assert data["collection"] == expected_coll
        explain_json_str = json.dumps(data["explain"])
        assert "IXSCAN" in explain_json_str, f"IXSCAN not found in {name} explain plan"
        assert "COLLSCAN" not in explain_json_str, f"COLLSCAN found in {name} explain plan!"
