from database import get_connection
from config import SAMPLE_SIZE

def function_statistics():

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

    df = con.execute(f"""
        SELECT

            funcName,

            COUNT(*)                AS invocations,

            COUNT(DISTINCT podID)   AS unique_pods,

            COUNT(DISTINCT userID)  AS unique_users,

            AVG(cpu_usage)          AS avg_cpu,

            AVG(memory_usage)       AS avg_memory,

            AVG(runtimeCost)        AS avg_runtime

        FROM {source}

        GROUP BY funcName

        ORDER BY invocations DESC

    """).df()

    con.close()

    return df
