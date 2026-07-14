#!/usr/bin/env python3

from pathlib import Path
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "database" / "huawei.duckdb"

LIMIT = 10000      # Tomorrow change to None

con = duckdb.connect(DB_PATH)

print("="*70)
print("Huawei Cloud FaaS Dataset Analysis")
print("="*70)

# ----------------------------------------------------------
# Base relation
# ----------------------------------------------------------

if LIMIT:
    source = f"(SELECT * FROM requests LIMIT {LIMIT})"
else:
    source = "requests"

# ----------------------------------------------------------
# Dataset summary
# ----------------------------------------------------------

print("\nDATASET SUMMARY")
print("-"*70)

summary = con.execute(f"""
SELECT
COUNT(*) AS requests,
COUNT(DISTINCT funcName),
COUNT(DISTINCT userID),
COUNT(DISTINCT podID),
COUNT(DISTINCT clusterName)
FROM {source}
""").fetchone()

print(f"Requests : {summary[0]:,}")
print(f"Functions: {summary[1]:,}")
print(f"Users    : {summary[2]:,}")
print(f"Pods     : {summary[3]:,}")
print(f"Clusters : {summary[4]}")

# ----------------------------------------------------------
# Region distribution
# ----------------------------------------------------------

print("\nREGION DISTRIBUTION")
print("-"*70)

rows = con.execute(f"""
SELECT
region,
COUNT(*)
FROM {source}
GROUP BY region
ORDER BY COUNT(*) DESC
""").fetchall()

for region,count in rows:
    print(f"{region:>3} : {count:,}")

# ----------------------------------------------------------
# Top Functions
# ----------------------------------------------------------

print("\nTOP 20 FUNCTIONS")
print("-"*70)

rows = con.execute(f"""
SELECT
funcName,
COUNT(*) AS invocations
FROM {source}
GROUP BY funcName
ORDER BY invocations DESC
LIMIT 20
""").fetchall()

for func,cnt in rows:
    print(f"{func:>6} : {cnt:,}")

# ----------------------------------------------------------
# Function reuse
# ----------------------------------------------------------

print("\nFUNCTION REUSE")
print("-"*70)

rows = con.execute(f"""
SELECT
AVG(requests_per_function),
MIN(requests_per_function),
MAX(requests_per_function)
FROM
(
SELECT
funcName,
COUNT(*) requests_per_function
FROM {source}
GROUP BY funcName
)
""").fetchone()

print(f"Average requests/function : {rows[0]:.2f}")
print(f"Minimum requests/function : {rows[1]}")
print(f"Maximum requests/function : {rows[2]}")

# ----------------------------------------------------------
# Pod reuse
# ----------------------------------------------------------

print("\nPOD REUSE")
print("-"*70)

rows = con.execute(f"""
SELECT
AVG(pods)
FROM
(
SELECT
funcName,
COUNT(DISTINCT podID) pods
FROM {source}
GROUP BY funcName
)
""").fetchone()

print(f"Average pods/function : {rows[0]:.2f}")

# ----------------------------------------------------------
# CPU statistics
# ----------------------------------------------------------

print("\nCPU USAGE")
print("-"*70)

rows = con.execute(f"""
SELECT
AVG(cpu_usage),
MIN(cpu_usage),
MAX(cpu_usage)
FROM {source}
""").fetchone()

print(f"Average CPU : {rows[0]:.4f}")
print(f"Minimum CPU : {rows[1]:.4f}")
print(f"Maximum CPU : {rows[2]:.4f}")

# ----------------------------------------------------------
# Memory statistics
# ----------------------------------------------------------

print("\nMEMORY USAGE")
print("-"*70)

rows = con.execute(f"""
SELECT
AVG(memory_usage),
MIN(memory_usage),
MAX(memory_usage)
FROM {source}
""").fetchone()

print(f"Average Memory : {rows[0]:.2f}")
print(f"Minimum Memory : {rows[1]:.2f}")
print(f"Maximum Memory : {rows[2]:.2f}")

print("\nDone.")

con.close()
