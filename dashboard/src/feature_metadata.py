"""Loads config/feature_metadata.json -- feature dtypes, valid categorical
values, and numeric ranges computed directly from the training data by
scripts/generate_feature_metadata.py. Nothing here is hand-invented."""

import json
from pathlib import Path

import streamlit as st

METADATA_PATH = Path(__file__).resolve().parent.parent / "config" / "feature_metadata.json"


@st.cache_data
def load_feature_metadata() -> dict:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Feature metadata not found at {METADATA_PATH}. Run "
            "`python scripts/generate_feature_metadata.py` from the repo root first."
        )
    with open(METADATA_PATH) as f:
        return json.load(f)


def get_feature(col: str) -> dict:
    meta = load_feature_metadata()
    if col not in meta["features"]:
        raise KeyError(f"Unknown feature: {col!r}")
    return meta["features"][col]


def feature_order() -> list[str]:
    return load_feature_metadata()["feature_order"]


def geo_cols() -> list[str]:
    return load_feature_metadata()["geo_cols"]


def numeric_cols() -> list[str]:
    return load_feature_metadata()["numeric_cols"]


def categorical_cols() -> list[str]:
    return load_feature_metadata()["categorical_cols"]


def binary_cols() -> list[str]:
    meta = load_feature_metadata()
    return meta["superstructure_cols"] + [meta["secondary_use_umbrella_col"]] + meta["secondary_use_sub_cols"]


def target_info() -> dict:
    return load_feature_metadata()["target"]


def default_building() -> dict:
    """A plausible baseline building: the most common category for each
    categorical/binary feature and the median for each numeric feature --
    used to seed the Predict and What-if pages before the user changes
    anything."""
    meta = load_feature_metadata()
    values = {}
    for col in meta["geo_cols"]:
        values[col] = None  # geo has no single sensible default -- user must pick
    for col in meta["numeric_cols"]:
        values[col] = round(meta["features"][col]["median"])
    for col in meta["categorical_cols"]:
        values[col] = meta["features"][col]["default"]
    for col in binary_cols():
        values[col] = meta["features"][col]["default"]
    return values
