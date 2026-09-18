"""SQL queries against the `buildings` table. All aggregation happens in
SQLite; Pandas only shapes the (already small) result sets returned here."""

import pandas as pd
import streamlit as st

from src.database import TABLE, run_query

ID_COL = "building_id"
TARGET_COL = "damage_grade"


# ---------------------------------------------------------------- schema / summary

@st.cache_data
def get_columns() -> pd.DataFrame:
    """PRAGMA table_info: name, sqlite type, etc."""
    return run_query(f"PRAGMA table_info({TABLE})")


@st.cache_data
def get_total_count() -> int:
    return int(run_query(f"SELECT COUNT(*) AS n FROM {TABLE}")["n"].iloc[0])


@st.cache_data
def get_distinct_values(column: str) -> list:
    df = run_query(
        f"SELECT DISTINCT {column} AS v FROM {TABLE} "
        f"WHERE {column} IS NOT NULL ORDER BY {column}"
    )
    return df["v"].tolist()


@st.cache_data
def get_min_max(column: str) -> tuple[float, float]:
    df = run_query(f"SELECT MIN({column}) AS lo, MAX({column}) AS hi FROM {TABLE}")
    return float(df["lo"].iloc[0]), float(df["hi"].iloc[0])


# ---------------------------------------------------------------- data quality

@st.cache_data
def get_missing_value_counts() -> pd.DataFrame:
    cols = get_columns()["name"].tolist()
    select_clause = ", ".join(f'SUM(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END) AS "{c}"' for c in cols)
    row = run_query(f"SELECT {select_clause} FROM {TABLE}").iloc[0]
    total = get_total_count()
    out = pd.DataFrame({"column": cols, "missing_count": [int(row[c]) for c in cols]})
    out["missing_pct"] = 100 * out["missing_count"] / total
    return out.sort_values("missing_count", ascending=False).reset_index(drop=True)


@st.cache_data
def get_duplicate_stats() -> dict:
    cols = [c for c in get_columns()["name"].tolist() if c != ID_COL]
    col_list = ", ".join(f'"{c}"' for c in cols)
    total = get_total_count()
    distinct_rows = int(
        run_query(f"SELECT COUNT(*) AS n FROM (SELECT DISTINCT {col_list} FROM {TABLE})")["n"].iloc[0]
    )
    distinct_ids = int(
        run_query(f"SELECT COUNT(DISTINCT {ID_COL}) AS n FROM {TABLE}")["n"].iloc[0]
    )
    return {
        "duplicate_rows": total - distinct_rows,
        "duplicate_building_ids": total - distinct_ids,
    }


# ---------------------------------------------------------------- full dataset

@st.cache_data
def get_full_dataset() -> pd.DataFrame:
    return run_query(f"SELECT * FROM {TABLE}")


# ---------------------------------------------------------------- feature analysis

@st.cache_data
def damage_distribution() -> pd.DataFrame:
    df = run_query(
        f"SELECT {TARGET_COL}, COUNT(*) AS buildings FROM {TABLE} "
        f"GROUP BY {TARGET_COL} ORDER BY {TARGET_COL}"
    )
    df["pct"] = 100 * df["buildings"] / df["buildings"].sum()
    return df


@st.cache_data
def feature_value_counts(feature: str) -> pd.DataFrame:
    df = run_query(
        f"SELECT {feature} AS value, COUNT(*) AS buildings FROM {TABLE} "
        f"GROUP BY {feature} ORDER BY buildings DESC"
    )
    df["pct"] = 100 * df["buildings"] / df["buildings"].sum()
    return df


@st.cache_data
def feature_vs_damage(feature: str) -> pd.DataFrame:
    return run_query(
        f"SELECT {feature} AS value, {TARGET_COL}, COUNT(*) AS buildings "
        f"FROM {TABLE} GROUP BY {feature}, {TARGET_COL}"
    )


@st.cache_data
def feature_and_target_raw(feature: str) -> pd.DataFrame:
    """Raw (feature, damage_grade) pairs - for numeric-vs-target box plots."""
    return run_query(f"SELECT {feature}, {TARGET_COL} FROM {TABLE}")


# ---------------------------------------------------------------- geography / sunburst

@st.cache_data
def geo1_summary() -> pd.DataFrame:
    return run_query(
        f"""
        SELECT geo_level_1_id, COUNT(*) AS buildings, AVG({TARGET_COL}) AS avg_damage_grade
        FROM {TABLE} GROUP BY geo_level_1_id ORDER BY buildings DESC
        """
    )


@st.cache_data
def geo2_summary_all() -> pd.DataFrame:
    """geo1+geo2 aggregates across the whole country (bounded: ~1.4k rows)."""
    return run_query(
        f"""
        SELECT geo_level_1_id, geo_level_2_id, COUNT(*) AS buildings,
               AVG({TARGET_COL}) AS avg_damage_grade
        FROM {TABLE} GROUP BY geo_level_1_id, geo_level_2_id
        """
    )


@st.cache_data
def geo3_summary_all() -> pd.DataFrame:
    """geo1+geo2+geo3 aggregates for the whole country (~11.6k rows)."""
    return run_query(
        f"""
        SELECT geo_level_1_id, geo_level_2_id, geo_level_3_id, COUNT(*) AS buildings,
               AVG({TARGET_COL}) AS avg_damage_grade
        FROM {TABLE} GROUP BY geo_level_1_id, geo_level_2_id, geo_level_3_id
        """
    )


# ---------------------------------------------------------------- cascading geo dropdowns

@st.cache_data
def geo1_options() -> list:
    """All geo_level_1_id values actually present in training data."""
    return run_query(
        f"SELECT DISTINCT geo_level_1_id AS v FROM {TABLE} ORDER BY geo_level_1_id"
    )["v"].tolist()


@st.cache_data
def geo2_options_for_geo1(geo_level_1_id: int) -> list:
    """geo_level_2_id values that actually co-occur with this geo_level_1_id
    in training data -- the real, observed hierarchy, not an invented one."""
    return run_query(
        f"SELECT DISTINCT geo_level_2_id AS v FROM {TABLE} "
        f"WHERE geo_level_1_id = ? ORDER BY geo_level_2_id",
        (geo_level_1_id,),
    )["v"].tolist()


@st.cache_data
def geo3_options_for_geo2(geo_level_1_id: int, geo_level_2_id: int) -> list:
    """geo_level_3_id values that actually co-occur with this
    (geo_level_1_id, geo_level_2_id) pair in training data."""
    return run_query(
        f"SELECT DISTINCT geo_level_3_id AS v FROM {TABLE} "
        f"WHERE geo_level_1_id = ? AND geo_level_2_id = ? ORDER BY geo_level_3_id",
        (geo_level_1_id, geo_level_2_id),
    )["v"].tolist()


# ---------------------------------------------------------------- filtered dataset explorer

# Whitelist of columns the filter panel is allowed to build WHERE clauses on
# -- values are always parameterized, but column names are still restricted
# to this fixed list rather than accepting arbitrary user-supplied text.
FILTERABLE_COLUMNS = {
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
}


def _build_where(filters: dict) -> tuple[str, list]:
    clauses, params = [], []
    for col, value in filters.items():
        if col not in FILTERABLE_COLUMNS or value in (None, "", "All"):
            continue
        clauses.append(f'"{col}" = ?')
        params.append(value)
    where_sql = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return where_sql, params


@st.cache_data
def filtered_summary(filters: dict) -> dict:
    """Total count + damage-grade distribution for rows matching `filters`
    (column -> value, only FILTERABLE_COLUMNS honored). Values are
    parameterized (not string-interpolated) to avoid SQL injection."""
    where_sql, params = _build_where(filters)
    total = run_query(f"SELECT COUNT(*) AS n FROM {TABLE}{where_sql}", params)["n"].iloc[0]
    dist = run_query(
        f"SELECT {TARGET_COL}, COUNT(*) AS buildings FROM {TABLE}{where_sql} "
        f"GROUP BY {TARGET_COL} ORDER BY {TARGET_COL}",
        params,
    )
    dist["pct"] = 100 * dist["buildings"] / total if total else 0.0
    return {"total": int(total), "distribution": dist}


@st.cache_data
def filtered_feature_vs_damage(feature: str, filters: dict) -> pd.DataFrame:
    """feature vs damage_grade crosstab counts, restricted to rows matching
    `filters`. `feature` must be a hardcoded column name from labels.py, not
    user-typed text."""
    where_sql, params = _build_where(filters)
    return run_query(
        f"SELECT {feature} AS value, {TARGET_COL}, COUNT(*) AS buildings "
        f"FROM {TABLE}{where_sql} GROUP BY {feature}, {TARGET_COL}",
        params,
    )
