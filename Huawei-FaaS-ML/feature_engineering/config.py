from pathlib import Path

# ``feature_engineering`` is a top-level package under Huawei-FaaS-ML.
# Keep its database location aligned with transformer_v2/config.py.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DB_PATH = PROJECT_ROOT / "database" / "huawei.duckdb"

SPARSE_OUTPUT_TABLE = "aggregated_requests"
DENSE_OUTPUT_TABLE = "aggregated_requests_dense"
