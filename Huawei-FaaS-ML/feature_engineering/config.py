from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DB_PATH = PROJECT_ROOT / "database" / "huawei.duckdb"

OUTPUT_TABLE = "aggregated_requests"
