#!/usr/bin/env python3

from pathlib import Path
import duckdb
import time
import traceback

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "extracted"
DB_PATH = PROJECT_ROOT / "database" / "huawei.duckdb"

print(f"Opening database:\n{DB_PATH}\n")

con = duckdb.connect(DB_PATH)

# Use all available vCPUs
con.execute("PRAGMA threads=2;")

# -------------------------------------------------------------------
# Metadata table
# -------------------------------------------------------------------

con.execute("""
CREATE TABLE IF NOT EXISTS import_log(
    csv_file TEXT PRIMARY KEY,
    region TEXT,
    row_count BIGINT,
    status TEXT,
    error TEXT,
    imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

# -------------------------------------------------------------------
# Discover CSV files
# -------------------------------------------------------------------

csv_files = sorted(DATA_DIR.rglob("*.csv"))

print(f"Discovered {len(csv_files)} CSV files.\n")

# -------------------------------------------------------------------
# Load already-imported files
# -------------------------------------------------------------------

already_imported = set(
    row[0]
    for row in con.execute("""
        SELECT csv_file
        FROM import_log
        WHERE status='SUCCESS'
    """).fetchall()
)

print(f"Already imported: {len(already_imported)} CSVs\n")

# -------------------------------------------------------------------
# Create requests table if it doesn't exist
# -------------------------------------------------------------------

exists = con.execute("""
SELECT COUNT(*)
FROM information_schema.tables
WHERE table_name='requests'
""").fetchone()[0]

if exists == 0:

    first_csv = csv_files[0]
    region = first_csv.parts[-3].split("_")[0]

    print("Creating requests table...\n")

    con.execute(f"""
        CREATE TABLE requests AS
        SELECT *,
               '{region}' AS region
        FROM read_csv_auto('{first_csv}');
    """)

    rows = con.execute(f"""
        SELECT COUNT(*)
        FROM read_csv_auto('{first_csv}')
    """).fetchone()[0]

    con.execute("""
        INSERT INTO import_log
        VALUES (?, ?, ?, 'SUCCESS', NULL, CURRENT_TIMESTAMP)
    """, [str(first_csv), region, rows])

    already_imported.add(str(first_csv))

# -------------------------------------------------------------------
# Import loop
# -------------------------------------------------------------------

success = 0
failed = 0

start = time.time()

for idx, csv in enumerate(csv_files, start=1):

    csv = csv.resolve()

    if str(csv) in already_imported:
        continue

    region = csv.parts[-3].split("_")[0]

    print(f"[{idx:4}/{len(csv_files)}] {csv.name}")

    try:

        row_count = con.execute(f"""
            SELECT COUNT(*)
            FROM read_csv_auto('{csv}')
        """).fetchone()[0]

        con.execute("BEGIN")

        con.execute(f"""
            INSERT INTO requests
            SELECT *,
                   '{region}' AS region
            FROM read_csv_auto('{csv}');
        """)

        con.execute("""
            INSERT INTO import_log
            VALUES (?, ?, ?, 'SUCCESS', NULL, CURRENT_TIMESTAMP)
        """, [str(csv), region, row_count])

        con.execute("COMMIT")

        already_imported.add(str(csv))
        success += 1

    except Exception as e:

        con.execute("ROLLBACK")

        error = str(e)[:500]

        print(f"   ERROR: {error}")

        con.execute("""
            INSERT OR REPLACE INTO import_log
            VALUES (?, ?, ?, 'FAILED', ?, CURRENT_TIMESTAMP)
        """, [str(csv), region, 0, error])

        failed += 1

print("\n========================================")
print("IMPORT FINISHED")
print("========================================")

print(f"Successful CSVs : {success}")
print(f"Failed CSVs     : {failed}")

print("\nTotal imported CSVs:",
      con.execute("""
        SELECT COUNT(*)
        FROM import_log
        WHERE status='SUCCESS'
      """).fetchone()[0])

print("Total failed CSVs:",
      con.execute("""
        SELECT COUNT(*)
        FROM import_log
        WHERE status='FAILED'
      """).fetchone()[0])

print("Total rows:",
      con.execute("""
        SELECT COUNT(*)
        FROM requests
      """).fetchone()[0])

print(f"\nElapsed: {(time.time()-start)/60:.2f} minutes")

con.close()
