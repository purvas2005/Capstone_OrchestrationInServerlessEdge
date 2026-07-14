import duckdb

from config import DB_PATH


def get_connection():
    return duckdb.connect(str(DB_PATH))
