from db import get_connection
from config import SAMPLE_SIZE

def dataset_summary():

    con = get_connection()

    if SAMPLE_SIZE is None:
        source = "requests"
    else:
        source = f"""
        (
            SELECT *
            FROM requests
            USING SAMPLE {SAMPLE_SIZE} ROWS
        )
        """

    row = con.execute(f"""
        SELECT
            COUNT(*),
            COUNT(DISTINCT funcName),
            COUNT(DISTINCT userID),
            COUNT(DISTINCT podID),
            COUNT(DISTINCT clusterName)
        FROM {source}
    """).fetchone()

    con.close()

    return {
        "requests": row[0],
        "functions": row[1],
        "users": row[2],
        "pods": row[3],
        "clusters": row[4],
    }
