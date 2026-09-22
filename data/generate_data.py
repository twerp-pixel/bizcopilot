"""
Generates a synthetic retail dataset for BizCopilot:
- sales, inventory, promotions tables in SQLite
- a small corpus of "analyst notes" / news docs (.txt) in data/docs/

Designed so that a handful of anomalies are INTENTIONALLY injected and
explained by specific docs, so the eval set has known ground truth
(useful because it's a portfolio project you need to defend in an interview).

Run: python3 generate_data.py
Requires: stdlib only (random, sqlite3, json) - no installs needed.
"""
import sqlite3
import random
import json
import os

random.seed(42)

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "bizcopilot.db")
DOCS_DIR = os.path.join(HERE, "docs")

REGIONS = ["Midwest", "Northeast", "South", "West"]
STORES = {r: [f"{r[:2].upper()}-{i:02d}" for i in range(1, 4)] for r in REGIONS}
SKUS = [f"SKU-{1000+i}" for i in range(20)]
WEEKS = list(range(1, 13))  # 12 week history

# ---- Ground-truth anomaly scenarios we will inject and later use for eval ----
SCENARIOS = [
    {
        "sku": "SKU-1004",
        "region": "Midwest",
        "week": 9,
        "type": "competitor_promo",
        "drop_pct": 0.32,
        "doc_id": "note_001",
    },
    {
        "sku": "SKU-1011",
        "region": "West",
        "week": 7,
        "type": "stockout",
        "drop_pct": 0.55,
        "doc_id": "note_002",
    },
    {
        "sku": "SKU-1017",
        "region": "South",
        "week": 10,
        "type": "quality_recall",
        "drop_pct": 0.70,
        "doc_id": "note_003",
    },
]
SCENARIO_INDEX = {(s["sku"], s["region"], s["week"]): s for s in SCENARIOS}


def gen_sales_inventory_promo():
    sales_rows = []
    inv_rows = []
    promo_rows = []

    for sku in SKUS:
        base_units = random.randint(80, 260)
        base_price = round(random.uniform(8, 40), 2)
        for region in REGIONS:
            for store in STORES[region]:
                on_hand = random.randint(300, 900)
                for week in WEEKS:
                    weekly_units = max(5, int(base_units * random.uniform(0.85, 1.15)))
                    promo_flag = 1 if random.random() < 0.15 else 0
                    if promo_flag:
                        weekly_units = int(weekly_units * random.uniform(1.2, 1.6))

                    scenario = SCENARIO_INDEX.get((sku, region, week))
                    if scenario:
                        weekly_units = int(weekly_units * (1 - scenario["drop_pct"]))

                    revenue = round(weekly_units * base_price, 2)
                    on_hand = max(0, on_hand - weekly_units + random.randint(-20, 60))

                    sales_rows.append((sku, store, region, week, weekly_units, revenue))
                    inv_rows.append((sku, store, region, week, on_hand))
                    promo_rows.append((sku, store, region, week, promo_flag))

    return sales_rows, inv_rows, promo_rows


def build_db():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE sales (
            sku TEXT, store TEXT, region TEXT, week INTEGER,
            units INTEGER, revenue REAL
        )
    """)
    cur.execute("""
        CREATE TABLE inventory (
            sku TEXT, store TEXT, region TEXT, week INTEGER,
            on_hand INTEGER
        )
    """)
    cur.execute("""
        CREATE TABLE promotions (
            sku TEXT, store TEXT, region TEXT, week INTEGER,
            promo_flag INTEGER
        )
    """)

    sales_rows, inv_rows, promo_rows = gen_sales_inventory_promo()
    cur.executemany("INSERT INTO sales VALUES (?,?,?,?,?,?)", sales_rows)
    cur.executemany("INSERT INTO inventory VALUES (?,?,?,?,?)", inv_rows)
    cur.executemany("INSERT INTO promotions VALUES (?,?,?,?,?)", promo_rows)

    conn.commit()
    conn.close()
    print(f"DB built at {DB_PATH} ({len(sales_rows)} sales rows)")


DOCS = {
    "note_001": (
        "Competitor Watch - Week 9 Midwest",
        "A major competitor launched an aggressive 25% off promotion on "
        "comparable products across Midwest stores starting week 9. Field "
        "reps report noticeable foot traffic shift for SKU-1004 in that "
        "region during the promo window."
    ),
    "note_002": (
        "Supply Chain Alert - Week 7 West Region",
        "SKU-1011 experienced a distribution center delay affecting West "
        "region stores in week 7, resulting in partial stockouts at "
        "several locations for several days that week."
    ),
    "note_003": (
        "Quality Notice - Week 10 South Region",
        "A voluntary quality review was initiated for a specific batch of "
        "SKU-1017 in week 10, affecting South region stores. Some stores "
        "pulled affected lots from shelves pending review, reducing "
        "available units for sale."
    ),
    "note_004": (
        "General Market Note - Q3",
        "Overall category demand has been broadly stable this quarter with "
        "typical week-to-week promotional variance of 10-20% in categories "
        "running periodic promotions."
    ),
    "note_005": (
        "Promo Calendar Overview",
        "Internal promotions are scheduled at roughly 15% frequency across "
        "SKUs and regions, typically producing a 20-60% uplift in the "
        "promoted week."
    ),
    "note_006": (
        "Regional Performance Summary",
        "Northeast and South regions have shown the most stable "
        "week-over-week sales patterns this quarter relative to Midwest "
        "and West, which have seen more promotional and competitive "
        "activity."
    ),
    "note_007": (
        "Vendor Update - SKU-1004",
        "Vendor confirmed no supply issues for SKU-1004 this quarter; any "
        "demand softness is attributed to external market factors rather "
        "than availability."
    ),
    "note_008": (
        "Store Ops Note - West Region",
        "West region distribution center delays in week 7 were resolved by "
        "week 8, with inventory levels returning to normal shortly after."
    ),
}


def build_docs():
    os.makedirs(DOCS_DIR, exist_ok=True)
    manifest = []
    for doc_id, (title, text) in DOCS.items():
        path = os.path.join(DOCS_DIR, f"{doc_id}.txt")
        with open(path, "w") as f:
            f.write(f"{title}\n\n{text}\n")
        manifest.append({"doc_id": doc_id, "title": title, "path": f"docs/{doc_id}.txt"})

    with open(os.path.join(HERE, "docs_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"{len(DOCS)} docs written to {DOCS_DIR}")


def save_scenarios():
    with open(os.path.join(HERE, "scenarios.json"), "w") as f:
        json.dump(SCENARIOS, f, indent=2)


if __name__ == "__main__":
    build_db()
    build_docs()
    save_scenarios()
    print("Done. Ground-truth scenarios saved to scenarios.json (for eval set construction).")
