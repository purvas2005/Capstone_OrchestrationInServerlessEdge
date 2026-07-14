from database import get_connection
import time

SOURCE = "aggregated_requests"
METADATA = "function_metadata"
TARGET = "feature_table"

con = get_connection()

print("=" * 70)
print("BUILDING FEATURE TABLE")
print("=" * 70)

start = time.time()

con.execute(f"DROP TABLE IF EXISTS {TARGET};")

con.execute(f"""
CREATE TABLE {TARGET} AS

WITH base AS (

SELECT

    *,

    ------------------------------------------------------------
    -- Lag Features
    ------------------------------------------------------------
    LAG(requests,1) OVER w AS lag_1,
    LAG(requests,5) OVER w AS lag_5,
    LAG(requests,15) OVER w AS lag_15,
    LAG(requests,30) OVER w AS lag_30,

    ------------------------------------------------------------
    -- Rolling Means
    ------------------------------------------------------------
    AVG(requests) OVER (
        PARTITION BY region, clusterName, funcName
        ORDER BY minute
        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    ) AS rolling_mean_5,

    AVG(requests) OVER (
        PARTITION BY region, clusterName, funcName
        ORDER BY minute
        ROWS BETWEEN 14 PRECEDING AND CURRENT ROW
    ) AS rolling_mean_15,

    AVG(requests) OVER (
        PARTITION BY region, clusterName, funcName
        ORDER BY minute
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) AS rolling_mean_30,

    ------------------------------------------------------------
    -- Rolling Standard Deviation
    ------------------------------------------------------------
    STDDEV_SAMP(requests) OVER (
        PARTITION BY region, clusterName, funcName
        ORDER BY minute
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) AS rolling_std,

    ------------------------------------------------------------
    -- Rolling Maximum
    ------------------------------------------------------------
    MAX(requests) OVER (
        PARTITION BY region, clusterName, funcName
        ORDER BY minute
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) AS rolling_max,

    ------------------------------------------------------------
    -- Rolling Minimum
    ------------------------------------------------------------
    MIN(requests) OVER (
        PARTITION BY region, clusterName, funcName
        ORDER BY minute
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) AS rolling_min,

    ------------------------------------------------------------
    -- Rolling Median
    ------------------------------------------------------------
    MEDIAN(requests) OVER (
        PARTITION BY region, clusterName, funcName
        ORDER BY minute
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) AS rolling_median,

    ------------------------------------------------------------
    -- Warm Capacity
    ------------------------------------------------------------
    QUANTILE_CONT(requests,0.90) OVER (
        PARTITION BY region, clusterName, funcName
        ORDER BY minute
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) AS warm_capacity

FROM {SOURCE}

WINDOW w AS (
    PARTITION BY region, clusterName, funcName
    ORDER BY minute
)

)

SELECT

    ------------------------------------------------------------
    -- Original Features
    ------------------------------------------------------------
    base.*,

    ------------------------------------------------------------
    -- Function Metadata
    ------------------------------------------------------------
    fm.category,
    fm.stability,
    fm.total_requests,

    ------------------------------------------------------------
    -- Trend
    ------------------------------------------------------------
    (base.requests - COALESCE(base.lag_1, base.requests))
        AS trend,

    ------------------------------------------------------------
    -- Growth Rate
    ------------------------------------------------------------
    CASE
        WHEN COALESCE(base.lag_1,0)=0
            THEN 0
        ELSE
            (base.requests-base.lag_1)/base.lag_1
    END
        AS growth_rate,

    ------------------------------------------------------------
    -- Coefficient of Variation
    ------------------------------------------------------------
    CASE
        WHEN base.rolling_mean_30=0
            THEN 0
        ELSE
            base.rolling_std/base.rolling_mean_30
    END
        AS coeff_variation,

    ------------------------------------------------------------
    -- Resource Efficiency
    ------------------------------------------------------------
    base.avg_cpu/NULLIF(base.requests,0)
        AS cpu_per_request,

    base.avg_memory/NULLIF(base.requests,0)
        AS memory_per_request,

    base.avg_runtime/NULLIF(base.requests,0)
        AS runtime_per_request,

    ------------------------------------------------------------
    -- Warm Ratio
    ------------------------------------------------------------
    CASE
        WHEN base.warm_capacity=0
            THEN 0
        ELSE
            base.requests/base.warm_capacity
    END
        AS warm_ratio,

    ------------------------------------------------------------
    -- Time Features
    ------------------------------------------------------------
    (base.minute % 60)
        AS minute_of_hour,

    ((base.minute/60)%24)
        AS hour_of_day,

    SIN(
        2*PI()*((base.minute/60)%24)/24.0
    )
        AS hour_sin,

    COS(
        2*PI()*((base.minute/60)%24)/24.0
    )
        AS hour_cos,

    SIN(
        2*PI()*(base.minute%60)/60.0
    )
        AS minute_sin,

    COS(
        2*PI()*(base.minute%60)/60.0
    )
        AS minute_cos

FROM base

LEFT JOIN {METADATA} fm
ON base.funcName = fm.funcName

ORDER BY

    base.minute,
    base.region,
    base.clusterName,
    base.funcName;

""")

elapsed = time.time() - start

print()
print(f"Completed in {elapsed/60:.2f} minutes")
print()

rows = con.execute(f"""
SELECT COUNT(*)
FROM {TARGET}
""").fetchone()[0]

print("Rows:", rows)
print()

print(
    con.execute(f"""
SELECT *
FROM {TARGET}
LIMIT 5
""").df()
)

print()

print("Columns")

cols = con.execute(f"""
DESCRIBE {TARGET}
""").fetchall()

for c in cols:
    print(c[0])

con.execute("CHECKPOINT;")

print()
print("Database checkpoint completed.")
