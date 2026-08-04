import duckdb

try:
    from .config import DB_PATH
except ImportError:  # Supports ``python feature_engineering/<script>.py``.
    from config import DB_PATH


def get_connection():
    return duckdb.connect(str(DB_PATH))
