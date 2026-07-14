#!/usr/bin/env python3

from pathlib import Path
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ======================================================
# CONFIGURATION
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DB_PATH = PROJECT_ROOT / "database" / "huawei.duckdb"

RESULTS = PROJECT_ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"

TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# None = Full Dataset
SAMPLE_SIZE = None

# ======================================================
# DATABASE
# ======================================================

con = duckdb.connect(DB_PATH)
con.execute("PRAGMA threads=2")

if SAMPLE_SIZE is None:
    con.execute("""
        CREATE OR REPLACE TEMP VIEW source AS
        SELECT * FROM requests
    """)
else:
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW source AS
        SELECT *
        FROM requests
        USING SAMPLE {SAMPLE_SIZE} ROWS
    """)

report = []

def heading(title):
    print("\n" + "="*70)
    print(title)
    print("="*70)
    report.append("\n" + title)

# ======================================================
# DATASET SUMMARY
# ======================================================

heading("DATASET SUMMARY")

summary = con.execute("""

SELECT

COUNT(*),

COUNT(DISTINCT funcName),

COUNT(DISTINCT userID),

COUNT(DISTINCT podID),

COUNT(DISTINCT clusterName),

COUNT(DISTINCT region),

AVG(cpu_usage),

AVG(memory_usage),

AVG(runtimeCost),

MIN(time_worker),

MAX(time_worker)

FROM source

""").fetchone()

labels = [
"Requests",
"Functions",
"Users",
"Pods",
"Clusters",
"Regions",
"Average CPU",
"Average Memory",
"Average Runtime",
"Start Time",
"End Time"
]

for l,v in zip(labels,summary):
    print(f"{l:25}: {v}")
    report.append(f"{l}: {v}")

pd.DataFrame({
    "Metric":labels,
    "Value":summary
}).to_csv(TABLES/"dataset_summary.csv",index=False)

# ======================================================
# REGION STATISTICS
# ======================================================

heading("REGION STATISTICS")

regions = con.execute("""

SELECT

region,

COUNT(*) requests,

AVG(cpu_usage) avg_cpu,

AVG(memory_usage) avg_memory,

AVG(runtimeCost) avg_runtime

FROM source

GROUP BY region

ORDER BY requests DESC

""").df()

print(regions)

regions.to_csv(TABLES/"region_statistics.csv",index=False)

# ======================================================
# FUNCTION STATISTICS
# ======================================================

heading("FUNCTION STATISTICS")

functions = con.execute("""

SELECT

funcName,

COUNT(*) invocations,

COUNT(DISTINCT podID) pods,

COUNT(DISTINCT userID) users,

AVG(cpu_usage) avg_cpu,

AVG(memory_usage) avg_memory,

AVG(runtimeCost) avg_runtime

FROM source

GROUP BY funcName

ORDER BY invocations DESC

""").df()

print(functions.head(20))

functions.to_csv(TABLES/"function_statistics.csv",index=False)

# ======================================================
# FUNCTION CLASSIFICATION
# ======================================================

heading("FUNCTION CLASSIFICATION")

q20 = functions["invocations"].quantile(0.20)
q80 = functions["invocations"].quantile(0.80)

conditions = [

    functions["invocations"] <= q20,

    functions["invocations"] >= q80

]

choices = [

    "Rare",

    "Hot"

]

functions["category"] = np.select(
    conditions,
    choices,
    default="Normal"
)

functions.to_csv(
    TABLES/"function_classification.csv",
    index=False
)

print(
    functions["category"].value_counts()
)

report.append(
    str(functions["category"].value_counts())
)

# ======================================================
# FUNCTION REUSE
# ======================================================

heading("FUNCTION REUSE")

functions["requests_per_pod"] = (
    functions["invocations"] / functions["pods"]
)

functions["requests_per_user"] = (
    functions["invocations"] / functions["users"]
)

functions.to_csv(
    TABLES/"function_reuse.csv",
    index=False
)

print()

print("Average requests/pod:",
      round(functions["requests_per_pod"].mean(),2))

print("Maximum requests/pod:",
      round(functions["requests_per_pod"].max(),2))

print("Average requests/user:",
      round(functions["requests_per_user"].mean(),2))

report.append(
    f"Average Requests/Pod: {functions['requests_per_pod'].mean()}"
)

# ======================================================
# WORKLOAD SKEW
# ======================================================

heading("WORKLOAD SKEW")

inv = np.sort(functions["invocations"].values)

n = len(inv)

cum = np.cumsum(inv)

lorenz = cum / cum[-1]

lorenz = np.insert(lorenz,0,0)

x = np.linspace(0,1,len(lorenz))

gini = 1 - 2*np.trapezoid(lorenz, x)

print(f"Gini Coefficient : {gini:.4f}")

report.append(f"Gini: {gini}")

top1 = int(max(1,n*0.01))
top5 = int(max(1,n*0.05))
top10 = int(max(1,n*0.10))

total = functions["invocations"].sum()

functions_sorted = functions.sort_values(
    "invocations",
    ascending=False
)

def pct(rows):
    return (
        rows["invocations"].sum()/total
    )*100

print(f"Top 1% : {pct(functions_sorted.head(top1)):.2f}%")

print(f"Top 5% : {pct(functions_sorted.head(top5)):.2f}%")

print(f"Top10% : {pct(functions_sorted.head(top10)):.2f}%")


# ======================================================
# TEMPORAL HOTSPOTS
# ======================================================

heading("TEMPORAL HOTSPOTS")

hotspots = con.execute("""

SELECT

CAST(FLOOR(time_worker/60) AS INTEGER) AS minute,

COUNT(*) AS requests,

COUNT(DISTINCT funcName) AS active_functions,

COUNT(DISTINCT podID) AS active_pods,

AVG(cpu_usage) AS avg_cpu,

AVG(memory_usage) AS avg_memory

FROM source

GROUP BY minute

ORDER BY minute

""").df()

hotspots.to_csv(
    TABLES/"hotspot_statistics.csv",
    index=False
)

print(hotspots.head())

plt.figure(figsize=(12,5))

plt.plot(
    hotspots["minute"],
    hotspots["requests"]
)

plt.xlabel("Minute")

plt.ylabel("Requests")

plt.title("Requests Per Minute")

plt.tight_layout()

plt.savefig(
    FIGURES/"hotspot_timeline.png"
)

plt.close()

report.append(
    f"Peak Requests/Minute: {hotspots['requests'].max()}"
)


# ======================================================
# STABLE VS NOISY
# ======================================================

heading("STABLE VS NOISY")

variation = con.execute("""

WITH minute_counts AS (

SELECT

funcName,

CAST(FLOOR(time_worker/60) AS INTEGER) AS minute,

COUNT(*) AS requests

FROM source

GROUP BY funcName, minute

)

SELECT

funcName,

AVG(requests) AS mean_requests,

STDDEV(requests) AS std_requests

FROM minute_counts

GROUP BY funcName

""").df()

variation["cv"] = (
    variation["std_requests"] /
    variation["mean_requests"]
)

threshold = variation["cv"].median()

variation["stability"] = np.where(

    variation["cv"] <= threshold,

    "Stable",

    "Noisy"

)

variation.to_csv(

    TABLES/"function_variation.csv",

    index=False

)

print(

variation["stability"].value_counts()

)

report.append(

str(

variation["stability"].value_counts()

)

)

# ======================================================
# PLOTS
# ======================================================

heading("GENERATING FIGURES")

plt.figure(figsize=(8,5))
plt.plot(functions_sorted["invocations"].values)
plt.xlabel("Function Rank")
plt.ylabel("Invocations")
plt.title("Function Invocation Distribution")
plt.tight_layout()
plt.savefig(FIGURES/"function_distribution.png")
plt.close()

plt.figure(figsize=(6,6))
plt.plot(x,lorenz,label="Lorenz Curve")
plt.plot([0,1],[0,1],'--')
plt.xlabel("Functions")
plt.ylabel("Cumulative Requests")
plt.title("Lorenz Curve")
plt.tight_layout()
plt.savefig(FIGURES/"lorenz_curve.png")
plt.close()

plt.figure(figsize=(8,5))
plt.hist(functions["requests_per_pod"],bins=30)
plt.xlabel("Requests per Pod")
plt.ylabel("Functions")
plt.title("Function Reuse Distribution")
plt.tight_layout()
plt.savefig(FIGURES/"function_reuse.png")
plt.close()


plt.figure(figsize=(8,5))

plt.hist(
    variation["cv"].dropna(),
    bins=30
)

plt.xlabel("Coefficient of Variation")

plt.ylabel("Functions")

plt.title("Function Stability")

plt.tight_layout()

plt.savefig(
    FIGURES/"stability_distribution.png"
)

plt.close()


counts = functions["category"].value_counts()

plt.figure(figsize=(6,6))

plt.pie(

counts,

labels=counts.index,

autopct="%1.1f%%"

)

plt.title("Function Categories")

plt.savefig(
    FIGURES/"function_categories.png"
)

plt.close()
print("Figures saved.")

# ======================================================
# REPORT
# ======================================================

with open(RESULTS/"report.txt","w") as f:
    f.write("\n".join(report))

con.close()

print("\nAnalysis complete.")
