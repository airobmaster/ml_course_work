"""One-time ingestion: CSV -> SQLite.

Joins train_values.csv + train_labels.csv on building_id, writes the
result to dashboard/data/earthquake.db, and indexes the columns the
dashboard filters on.

Usage: python dashboard/scripts/build_db.py
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

DASHBOARD_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = DASHBOARD_DIR.parent
DATASETS_DIR = ROOT_DIR / "datasets"
DB_DIR = DASHBOARD_DIR / "data"
DB_PATH = DB_DIR / "earthquake.db"
TABLE = "buildings"

INDEXED_COLUMNS = [
    "geo_level_1_id",
    "geo_level_2_id",
    "geo_level_3_id",
    "foundation_type",
    "roof_type",
    "ground_floor_type",
    "other_floor_type",
    "land_surface_condition",
    "position",
    "plan_configuration",
    "legal_ownership_status",
    "damage_grade",
]


def main() -> None:
    values_path = DATASETS_DIR / "train_values.csv"
    labels_path = DATASETS_DIR / "train_labels.csv"

    if not values_path.exists() or not labels_path.exists():
        sys.exit(f"Expected CSVs not found in {DATASETS_DIR}")

    print(f"Reading {values_path.name} and {labels_path.name} ...")
    values = pd.read_csv(values_path)
    labels = pd.read_csv(labels_path)
    df = values.merge(labels, on="building_id", how="inner")
    print(f"Joined dataset: {len(df):,} rows, {len(df.columns)} columns")

    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    df.to_sql(TABLE, conn, if_exists="replace", index=False)

    cur = conn.cursor()
    cur.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS idx_building_id ON {TABLE}(building_id)")
    for col in INDEXED_COLUMNS:
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{col} ON {TABLE}({col})")
    conn.commit()
    conn.close()

    print(f"Wrote {DB_PATH} ({DB_PATH.stat().st_size / 1e6:.1f} MB) with {len(INDEXED_COLUMNS)} indexes")


if __name__ == "__main__":
    main()
