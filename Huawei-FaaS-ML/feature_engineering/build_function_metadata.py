from database import get_connection
import time

SOURCE = "aggregated_requests"
TARGET = "function_metadata"

con = get_connection()

print("=" * 70)
print("BUILDING FUNCTION METADATA")
print("=" * 70)

start = time.time()

con.execute(f"DROP TABLE IF EXISTS {TARGET};")

con.execute(f"""
CREATE TABLE {TARGET} AS

WITH stats AS (

SELECT

    funcName,

    SUM(requests) AS total_requests,

    AVG(avg_cpu) AS avg_cpu,

    AVG(avg_memory) AS avg_memory,

    AVG(avg_runtime) AS avg_runtime,

    STDDEV_SAMP(requests) AS std_requests,

    AVG(requests) AS mean_requests

FROM {SOURCE}

GROUP BY funcName

),

classified AS (

SELECT

*,

CASE

WHEN total_requests >= (
SELECT quantile_cont(total_requests,0.80) FROM stats
)
THEN 'Hot'

WHEN total_requests <= (
SELECT quantile_cont(total_requests,0.25) FROM stats
)
THEN 'Rare'

ELSE 'Normal'

END AS category,

CASE

WHEN mean_requests=0 THEN 'Noisy'

WHEN std_requests/mean_requests > 1
THEN 'Noisy'

ELSE 'Stable'

END AS stability

FROM stats

)

SELECT *

FROM classified;
""")

elapsed = time.time()-start

print(f"\nCompleted in {elapsed:.2f} seconds\n")

print(con.execute("""
SELECT category,COUNT(*)
FROM function_metadata
GROUP BY category
""").df())

print()

print(con.execute("""
SELECT stability,COUNT(*)
FROM function_metadata
GROUP BY stability
""").df())
