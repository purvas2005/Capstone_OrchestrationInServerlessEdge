import duckdb
from config import DB_PATH

def get_connection():
    con = duckdb.connect(DB_PATH)
    con.execute("PRAGMA threads=2;")
    return con
