# DECISIONS.md  (about 150-200 words in total - plain language, your own words)

1. **Indexes.**
   - `employees`: unique `{"emp_code": 1}` ensures customer business codes never duplicate; `{"department": 1, "joined_on": 1}` and `{"joined_on": 1}` accelerate department headcount filters.
   - `attendance_logs`: unique `{"emp_code": 1, "date": 1}` guarantees one record per employee-day and resolves punch-in race conditions; `{"date": -1, "emp_code": 1}` backs attendance pagination and range queries; `{"emp_code": 1, "punch_out": 1, "punch_in": -1}` enables sub-millisecond lookups for open shifts; `{"date": 1, "late_minutes": -1}` backs monthly leaderboard aggregation.
   - *Considered and rejected:* an index on `status` alone, due to low cardinality (5 values) and negligible selectivity compared to composite date indexes.

2. **Punch-in race.**
   Both requests validate the employee and compute the same attendance date. When both invoke `insert_one`, MongoDB's compound unique index `(emp_code, date)` guarantees atomic mutual exclusion: exactly one document is inserted (201 Created). The second concurrent insert encounters a `DuplicateKeyError`, which our API intercepts and returns as a 409 Conflict without relying on unreliable check-then-insert application logic.

3. **Ties.**
   We rank late minutes using MongoDB's `$rank` window function, producing standard competition ranks (e.g., 1, 2, 2, 4). Filtering (`rank <= limit`) is applied after ranking, with rows sorted by total late minutes descending then `emp_code` ascending. Tied employees at the cutoff share the same rank and are all returned, purposefully exceeding `limit` rows.

4. **Headcount.**
   The pipeline originates on the `employees` collection matching `joined_on <= month_end`, performing a `$lookup` into `attendance_logs` and an `$unwind` with `preserveNullAndEmptyArrays: true`. Employees with zero attendance records are preserved with null logs, correctly contributing to `headcount = {$size: "$employees"}` while contributing 0 to present days.

5. **One thing you would change for 100x the data.**
   Pre-aggregate attendance metrics into daily and monthly summary collections via change streams or asynchronous event workers, turning multi-stage cross-collection aggregations into indexed primary key lookups.
