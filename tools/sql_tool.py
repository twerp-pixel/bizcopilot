"""
SQL tool: gives the agent a safe, read-only way to query structured data.
Only SELECT statements are allowed. Returns rows as list[dict] so results
are easy to cite (row-level provenance) in the final memo.
"""
import sqlite3
import os
import re

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "bizcopilot.db")

SCHEMA_DESCRIPTION = """
Tables:
  sales(sku TEXT, store TEXT, region TEXT, week INTEGER, units INTEGER, revenue REAL)
  inventory(sku TEXT, store TEXT, region TEXT, week INTEGER, on_hand INTEGER)
  promotions(sku TEXT, store TEXT, region TEXT, week INTEGER, promo_flag INTEGER)

Weeks are integers 1-12 (a 12-week rolling window). Regions: Midwest, Northeast, South, West.
"""

_FORBIDDEN = re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|PRAGMA)\b", re.IGNORECASE)


class SQLToolError(Exception):
    pass


def run_sql(query: str, max_rows: int = 200):
    """Execute a read-only SQL query. Raises SQLToolError if the query looks unsafe."""
    normalized = query.strip().lower()

    if _FORBIDDEN.search(query):
        raise SQLToolError("Only read-only SELECT queries are permitted.")

    if not (
        normalized.startswith("select")
        or normalized.startswith("with")
    ):
        raise SQLToolError("Query must start with SELECT or WITH.")

    # URI mode=ro prevents writes even if a future query slips past the
    # lightweight SQL keyword checks below.
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(query)
        rows = [dict(r) for r in cur.fetchmany(max_rows)]
        return rows
    finally:
        conn.close()


def get_schema() -> str:
    return SCHEMA_DESCRIPTION


if __name__ == "__main__":
    print(get_schema())
    print(run_sql("SELECT sku, region, week, units FROM sales WHERE sku='SKU-1004' AND region='Midwest' ORDER BY week LIMIT 12"))
