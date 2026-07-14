from database import get_connection
from config import OUTPUT_TABLE
import time

con = get_connection()

print("=" * 70)
print("Creating Aggregated Dataset")
print("=" * 70)

start = time.time()

con.execute(f"""
DROP TABLE IF EXISTS {OUTPUT_TABLE};
""")

print("Building aggregated table...")

con.execute(f"""
CREATE TABLE {OUTPUT_TABLE} AS

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

ORDER BY

minute,
region,
funcName;
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
FROM {OUTPUT_TABLE}
""").fetchone()[0]

print("Rows:", rows)

print()

print(con.execute(f"""
SELECT *

FROM {OUTPUT_TABLE}

LIMIT 10
""").df())

print()

print(con.execute(f"""
SELECT

MIN(minute),
MAX(minute)

FROM {OUTPUT_TABLE}
""").fetchall())

print()

print(con.execute(f"""
SELECT

COUNT(DISTINCT funcName)

FROM {OUTPUT_TABLE}
""").fetchall())

print()

print(con.execute(f"""
SELECT

COUNT(DISTINCT region)

FROM {OUTPUT_TABLE}
""").fetchall())
