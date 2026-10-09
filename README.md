# Employee Attendance & Analytics API

A FastAPI and MongoDB backend service implementing employee attendance tracking (punch-in, punch-out, manual regularization with audit trails) and real-time HR analytics powered by MongoDB aggregation pipelines.

## Requirements
- Python 3.11+
- MongoDB 6.0+ (MongoDB 7.0 recommended)

## Setup & Running

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables (Optional):**
   ```bash
   export MONGO_URI="mongodb://localhost:27017"
   export MONGO_DB="attendance_db"
   ```
   *(Defaults to `mongodb://localhost:27017` and `attendance_db` if not set).*

4. **Seed Sample Data (Optional):**
   ```bash
   python sample_seed.py
   ```

5. **Start the API Server:**
   ```bash
   uvicorn app.main:app --port 8000
   ```

All database indexes (unique business keys and performance indexes) are initialized automatically and idempotently on application startup.

## Running Tests
```bash
pip install pytest httpx
pytest tests/ -v
```

## Documentation & Decisions
- **`REVIEW.md`**: Detailed review of the starter codebase defects, symptoms, and implemented fixes.
- **`DECISIONS.md`**: Architectural decisions regarding database indexes, concurrency handling, tie-breaking, headcount calculation, and 100x scaling strategy.
- **`openapi.yaml`**: Full API contract specification.
