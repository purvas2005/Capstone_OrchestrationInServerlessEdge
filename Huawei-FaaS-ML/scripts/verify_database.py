from pathlib import Path
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "database" / "huawei.duckdb"

con = duckdb.connect(DB_PATH)

print("=" * 60)
print("DATABASE SUMMARY")
print("=" * 60)

print("\nTables:")
print(con.execute("SHOW TABLES").fetchall())

print("\nTotal rows:")
print(con.execute("SELECT COUNT(*) FROM requests").fetchone()[0])

print("\nRows by region:")
for row in con.execute("""
SELECT region, COUNT(*)
FROM requests
GROUP BY region
ORDER BY region
""").fetchall():
    print(row)

print("\nUnique functions:")
print(con.execute("""
SELECT COUNT(DISTINCT funcName)
FROM requests
""").fetchone()[0])

print("\nUnique users:")
print(con.execute("""
SELECT COUNT(DISTINCT userID)
FROM requests
""").fetchone()[0])

print("\nUnique pods:")
print(con.execute("""
SELECT COUNT(DISTINCT podID)
FROM requests
""").fetchone()[0])

print("\nUnique clusters:")
print(con.execute("""
SELECT COUNT(DISTINCT clusterName)
FROM requests
""").fetchone()[0])

print("\nTime range:")
print(con.execute("""
SELECT
MIN(time_worker),
MAX(time_worker)
FROM requests
""").fetchone())

con.close()
