from pathlib import Path
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "database" / "huawei.duckdb"

con = duckdb.connect(DB_PATH)

print("\nTables:")
print(con.execute("SHOW TABLES").fetchall())

print("\nRow count:")
print(con.execute("SELECT COUNT(*) FROM requests").fetchall())

print("\nFirst five rows:")
rows = con.execute("SELECT * FROM requests LIMIT 5").fetchall()

for row in rows:
    print(row)

print("\nRegion counts:")
print(con.execute("""
SELECT region, COUNT(*)
FROM requests
GROUP BY region
""").fetchall())

con.close()
