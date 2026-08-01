try:
    from .database import get_connection
    from .config import DENSE_OUTPUT_TABLE, SPARSE_OUTPUT_TABLE
except ImportError:  # Supports direct execution from this directory.
    from database import get_connection
    from config import DENSE_OUTPUT_TABLE, SPARSE_OUTPUT_TABLE
import time

con = get_connection()

print("=" * 70)
print("Creating Aggregated Dataset")
print("=" * 70)

start = time.time()

con.execute(f"DROP TABLE IF EXISTS {SPARSE_OUTPUT_TABLE};")
con.execute(f"DROP TABLE IF EXISTS {DENSE_OUTPUT_TABLE};")

print("Building aggregated table...")

con.execute(f"""
CREATE TABLE {SPARSE_OUTPUT_TABLE} AS

SELECT

CAST(FLOOR(time_worker/60) AS INTEGER) AS minute,

region,

clusterName,

funcName,

COUNT(*) AS requests,

AVG(cpu_usage) AS avg_cpu,

AVG(memory_usage) AS avg_memory,

AVG(runtimeCost) AS avg_runtime,

AVG(requestBodySize) AS avg_request_size,

COUNT(DISTINCT podID) AS active_pods,

COUNT(DISTINCT userID) AS active_users

FROM requests

GROUP BY

minute,
region,
clusterName,
funcName

ORDER BY minute, region, clusterName, funcName;
""")

print("Expanding to a complete wall-clock minute calendar...")

# A missing event row is now represented explicitly as zero workload.  The
# global day bounds ensure every observed function group has the same set of
# minute timestamps, so a model horizon is genuinely H future minutes.
con.execute(f"""
CREATE TABLE {DENSE_OUTPUT_TABLE} AS
WITH
bounds AS (
    SELECT MIN(minute) AS first_minute, MAX(minute) AS last_minute
    FROM {SPARSE_OUTPUT_TABLE}
),
function_groups AS (
    SELECT DISTINCT region, clusterName, funcName
    FROM {SPARSE_OUTPUT_TABLE}
),
minute_calendar AS (
    SELECT CAST(r.minute AS INTEGER) AS minute
    FROM bounds,
    range(first_minute, last_minute + 1) AS r(minute)
)
SELECT
    groups.region,
    groups.clusterName,
    groups.funcName,
    calendar.minute,
    COALESCE(observed.requests, 0)::BIGINT AS requests,
    COALESCE(observed.avg_cpu, 0.0) AS avg_cpu,
    COALESCE(observed.avg_memory, 0.0) AS avg_memory,
    COALESCE(observed.avg_runtime, 0.0) AS avg_runtime,
    COALESCE(observed.avg_request_size, 0.0) AS avg_request_size,
    COALESCE(observed.active_pods, 0)::BIGINT AS active_pods,
    COALESCE(observed.active_users, 0)::BIGINT AS active_users
FROM function_groups AS groups
CROSS JOIN minute_calendar AS calendar
LEFT JOIN {SPARSE_OUTPUT_TABLE} AS observed
    ON observed.region = groups.region
    AND observed.clusterName = groups.clusterName
    AND observed.funcName = groups.funcName
    AND observed.minute = calendar.minute
ORDER BY groups.region, groups.clusterName, groups.funcName, calendar.minute;
""")

elapsed = time.time() - start

print()
print("Finished.")
print(f"Elapsed: {elapsed/60:.2f} minutes")



print()
print("=" * 70)
print("VERIFYING")
print("=" * 70)

rows = con.execute(f"""
SELECT COUNT(*)
FROM {DENSE_OUTPUT_TABLE}
""").fetchone()[0]

print("Rows:", rows)

print()

dense_rows, zero_rows = con.execute(f"""
SELECT COUNT(*), COUNT(*) FILTER (WHERE requests = 0)
FROM {DENSE_OUTPUT_TABLE}
""").fetchone()

expected_rows, invalid_groups = con.execute(f"""
WITH bounds AS (
    SELECT MIN(minute) AS first_minute, MAX(minute) AS last_minute
    FROM {DENSE_OUTPUT_TABLE}
),
per_group AS (
    SELECT
        region,
        clusterName,
        funcName,
        COUNT(*) AS minute_rows,
        MIN(minute) AS first_minute,
        MAX(minute) AS last_minute
    FROM {DENSE_OUTPUT_TABLE}
    GROUP BY region, clusterName, funcName
)
SELECT
    (SELECT COUNT(*) FROM per_group) *
        ((SELECT last_minute FROM bounds) - (SELECT first_minute FROM bounds) + 1),
    COUNT(*) FILTER (
        WHERE minute_rows <> (SELECT last_minute - first_minute + 1 FROM bounds)
        OR first_minute <> (SELECT first_minute FROM bounds)
        OR last_minute <> (SELECT last_minute FROM bounds)
    )
FROM per_group
""").fetchone()

print(f"Dense rows: {dense_rows:,}")
print(f"Zero-request minutes added: {zero_rows:,}")
print(f"Expected dense rows: {expected_rows:,}")
print(f"Groups with incomplete calendars: {invalid_groups:,}")

if dense_rows != expected_rows or invalid_groups != 0:
    raise RuntimeError("Dense calendar validation failed; feature engineering was stopped.")

print()

print(con.execute(f"""
SELECT *

FROM {DENSE_OUTPUT_TABLE}

LIMIT 10
""").df())

print()

print(con.execute(f"""
SELECT

MIN(minute),
MAX(minute)

FROM {DENSE_OUTPUT_TABLE}
""").fetchall())

print()

print(con.execute(f"""
SELECT

COUNT(DISTINCT funcName)

FROM {DENSE_OUTPUT_TABLE}
""").fetchall())

print()

print(con.execute(f"""
SELECT

COUNT(DISTINCT region)

FROM {DENSE_OUTPUT_TABLE}
""").fetchall())
