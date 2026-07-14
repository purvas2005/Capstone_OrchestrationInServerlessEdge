from pathlib import Path
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "database" / "huawei.duckdb"

con = duckdb.connect(DB_PATH)

print("Analyzing database...")
con.execute("ANALYZE requests;")

print("Checkpointing...")
con.execute("CHECKPOINT;")

print("Done.")

con.close()
