#!/usr/bin/env python
"""Generate config/feature_metadata.json from the actual training data plus
the project's own categorical code dictionary (dashboard/src/labels.py,
sourced from Categorical_Feature_Codes.md). Numeric ranges/means and
categorical value lists are computed directly from datasets/train_values.csv
-- nothing here is hand-invented.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dashboard"))

from src.labels import (  # noqa: E402
    CATEGORICAL_LABELS,
    FEATURE_DESCRIPTIONS,
    display_name,
)

DATASETS_DIR = ROOT / "datasets"
CONFIG_DIR = ROOT / "dashboard" / "config"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

GEO_COLS = ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]
NUMERIC_COLS = [
    "count_floors_pre_eq",
    "age",
    "area_percentage",
    "height_percentage",
    "count_families",
]
CATEGORICAL_COLS = list(CATEGORICAL_LABELS.keys())
SUPERSTRUCTURE_COLS = [
    "has_superstructure_adobe_mud",
    "has_superstructure_mud_mortar_stone",
    "has_superstructure_stone_flag",
    "has_superstructure_cement_mortar_stone",
    "has_superstructure_mud_mortar_brick",
    "has_superstructure_cement_mortar_brick",
    "has_superstructure_timber",
    "has_superstructure_bamboo",
    "has_superstructure_rc_non_engineered",
    "has_superstructure_rc_engineered",
    "has_superstructure_other",
]
SECONDARY_USE_SUB_COLS = [
    "has_secondary_use_agriculture",
    "has_secondary_use_hotel",
    "has_secondary_use_rental",
    "has_secondary_use_institution",
    "has_secondary_use_school",
    "has_secondary_use_industry",
    "has_secondary_use_health_post",
    "has_secondary_use_gov_office",
    "has_secondary_use_use_police",
    "has_secondary_use_other",
]
BINARY_COLS = SUPERSTRUCTURE_COLS + ["has_secondary_use"] + SECONDARY_USE_SUB_COLS


def main():
    train_values = pd.read_csv(DATASETS_DIR / "train_values.csv")
    train_labels = pd.read_csv(DATASETS_DIR / "train_labels.csv")

    features = {}

    for col in GEO_COLS:
        s = train_values[col]
        features[col] = {
            "name": col,
            "display_name": display_name(col),
            "dtype": "int",
            "feature_type": "geo",
            "description": FEATURE_DESCRIPTIONS.get(col, ""),
            "min": int(s.min()),
            "max": int(s.max()),
            "n_unique_in_train": int(s.nunique()),
        }

    for col in NUMERIC_COLS:
        s = train_values[col]
        features[col] = {
            "name": col,
            "display_name": display_name(col),
            "dtype": "int",
            "feature_type": "numeric",
            "description": FEATURE_DESCRIPTIONS.get(col, ""),
            "min": int(s.min()),
            "max": int(s.max()),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "p1": float(s.quantile(0.01)),
            "p99": float(s.quantile(0.99)),
        }

    for col in CATEGORICAL_COLS:
        s = train_values[col]
        counts = s.value_counts()
        values = [
            {
                "code": code,
                "label": CATEGORICAL_LABELS[col].get(code, code),
                "count": int(counts.get(code, 0)),
                "pct": float(100 * counts.get(code, 0) / len(s)),
            }
            for code in sorted(s.unique().tolist())
        ]
        features[col] = {
            "name": col,
            "display_name": display_name(col),
            "dtype": "category",
            "feature_type": "categorical",
            "description": FEATURE_DESCRIPTIONS.get(col, ""),
            "values": values,
            "default": counts.idxmax(),
        }

    for col in BINARY_COLS:
        s = train_values[col]
        prevalence = float(100 * s.mean())
        features[col] = {
            "name": col,
            "display_name": display_name(col),
            "dtype": "binary",
            "feature_type": "binary",
            "description": FEATURE_DESCRIPTIONS.get(col, ""),
            "values": [0, 1],
            "prevalence_pct": prevalence,
            "default": int(s.mode().iloc[0]),
        }

    target_counts = train_labels["damage_grade"].value_counts().sort_index()
    target = {
        "name": "damage_grade",
        "display_name": "Damage Grade",
        "classes": [int(c) for c in sorted(target_counts.index)],
        "class_labels": {
            "1": "Grade 1 - Low damage",
            "2": "Grade 2 - Medium damage",
            "3": "Grade 3 - Severe damage / destroyed",
        },
        "class_distribution": {
            str(int(k)): {"count": int(v), "pct": float(100 * v / target_counts.sum())}
            for k, v in target_counts.items()
        },
    }

    metadata = {
        "source": "Computed from datasets/train_values.csv and datasets/train_labels.csv (260,601 rows)",
        "id_col": "building_id",
        "target": target,
        "feature_order": GEO_COLS + NUMERIC_COLS + CATEGORICAL_COLS + SUPERSTRUCTURE_COLS
        + ["has_secondary_use"] + SECONDARY_USE_SUB_COLS,
        "geo_cols": GEO_COLS,
        "numeric_cols": NUMERIC_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "superstructure_cols": SUPERSTRUCTURE_COLS,
        "secondary_use_umbrella_col": "has_secondary_use",
        "secondary_use_sub_cols": SECONDARY_USE_SUB_COLS,
        "features": features,
    }

    out_path = CONFIG_DIR / "feature_metadata.json"
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved feature metadata to {out_path}")
    print(f"Total features described: {len(features)}")


if __name__ == "__main__":
    main()
