from database import get_connection
import time

SOURCE_TABLE = "aggregated_requests"
OUTPUT_TABLE = "aggregated_requests_features"

con = get_connection()

print("=" * 70)
print("GENERATING ROLLING FEATURES")
print("=" * 70)

start = time.time()

con.execute(f"DROP TABLE IF EXISTS {OUTPUT_TABLE};")

print("Computing rolling statistics...")

con.execute(f"""
CREATE TABLE {OUTPUT_TABLE} AS

WITH base AS (

SELECT
    *,

    -------------------------------------------------------------
    -- Previous minute workload
    -------------------------------------------------------------
    LAG(requests,1) OVER(
        PARTITION BY region, clusterName, funcName
        ORDER BY minute
    ) AS prev_requests,

    -------------------------------------------------------------
    -- Rolling Mean
    -------------------------------------------------------------
    AVG(requests) OVER(
        PARTITION BY region, clusterName, funcName
        ORDER BY minute
        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    ) AS rolling_mean_5,

    AVG(requests) OVER(
        PARTITION BY region, clusterName, funcName
        ORDER BY minute
        ROWS BETWEEN 14 PRECEDING AND CURRENT ROW
    ) AS rolling_mean_15,

    AVG(requests) OVER(
        PARTITION BY region, clusterName, funcName
        ORDER BY minute
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) AS rolling_mean_30,

    -------------------------------------------------------------
    -- Rolling Standard Deviation
    -------------------------------------------------------------
    STDDEV_SAMP(requests) OVER(
        PARTITION BY region, clusterName, funcName
        ORDER BY minute
        ROWS BETWEEN 14 PRECEDING AND CURRENT ROW
    ) AS rolling_std_15,

    STDDEV_SAMP(requests) OVER(
        PARTITION BY region, clusterName, funcName
        ORDER BY minute
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) AS rolling_std_30,

    -------------------------------------------------------------
    -- Exponential-style approximation
    -------------------------------------------------------------
    AVG(requests) OVER(
        PARTITION BY region, clusterName, funcName
        ORDER BY minute
        ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
    ) AS ema_10,

    -------------------------------------------------------------
    -- Warm Capacity
    -- 90th percentile over previous 30 minutes
    -------------------------------------------------------------
    QUANTILE_CONT(
        requests,
        0.90
    ) OVER(
        PARTITION BY region, clusterName, funcName
        ORDER BY minute
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) AS warm_capacity

FROM {SOURCE_TABLE}

)

SELECT

*,

-------------------------------------------------------------
-- Trend
-------------------------------------------------------------
(requests - COALESCE(prev_requests, requests)) AS trend,

-------------------------------------------------------------
-- Percent Growth
-------------------------------------------------------------
CASE

WHEN COALESCE(prev_requests,0)=0
THEN 0

ELSE
(requests-prev_requests)/prev_requests

END AS growth_rate

FROM base

ORDER BY

minute,
region,
clusterName,
funcName;
""")

elapsed = time.time() - start

print()
print(f"Finished in {elapsed/60:.2f} minutes.")

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

print("Columns")

cols = con.execute(f"""
DESCRIBE {OUTPUT_TABLE}
""").fetchall()

for c in cols:
    print(c[0])

con.execute("CHECKPOINT;")

print()
print("Database checkpoint completed.")
