import pandas as pd
import streamlit as st

from src import geo
from src import queries as q
from src.labels import (
    ANALYZABLE_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_DESCRIPTIONS,
    NUMERIC_FEATURES,
    code_to_label,
    display_name,
    feature_group,
)
from src.plotting import (
    damage_heatmap,
    geo_sunburst,
    numeric_box_by_damage,
    numeric_histogram,
    stacked_damage_bar,
    value_counts_bar,
)

st.title("📊 Dataset Explorer")
st.caption(
    "Historical analysis of the 260,601 labeled training buildings, backed by SQLite "
    "(`dashboard/data/earthquake.db`) with aggregation done in SQL, not by loading the "
    "full dataset into memory on every interaction. This is for exploring past damage "
    "patterns — not a source of inputs for new predictions (see Predict Damage)."
)

try:
    total = q.get_total_count()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

tab_overview, tab_feature, tab_geo, tab_filter = st.tabs(
    ["Overview", "Feature Analysis", "Geography", "Filtered Explorer"]
)

# ---------------------------------------------------------------- Overview
with tab_overview:
    columns_df = q.get_columns()
    total_cols = len(columns_df)
    feature_cols = [c for c in columns_df["name"] if c not in ("building_id", "damage_grade")]

    st.subheader("Dataset Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total buildings", f"{total:,}")
    c2.metric("Total columns", total_cols)
    c3.metric("Feature columns", len(feature_cols))
    c4.metric("Damage grade classes", 3)

    st.divider()
    st.subheader("Data Quality")
    dup = q.get_duplicate_stats()
    missing = q.get_missing_value_counts()
    total_missing_cells = int(missing["missing_count"].sum())

    q1, q2, q3 = st.columns(3)
    q1.metric("Duplicate rows", f"{dup['duplicate_rows']:,}")
    q2.metric("Duplicate building IDs", f"{dup['duplicate_building_ids']:,}")
    q3.metric("Missing cells (total)", f"{total_missing_cells:,}")

    if total_missing_cells == 0:
        st.success("No missing values in any column.")
    else:
        with st.expander("Missing values by column"):
            show = missing[missing["missing_count"] > 0].copy()
            show["missing_pct"] = show["missing_pct"].round(2)
            st.dataframe(show, width="stretch", hide_index=True)

    st.divider()
    st.subheader("Target Distribution")
    dist = q.damage_distribution()
    d1, d2, d3 = st.columns(3)
    for col_widget, (_, row) in zip((d1, d2, d3), dist.iterrows()):
        col_widget.metric(f"Grade {int(row['damage_grade'])}", f"{row['buildings']:,}", f"{row['pct']:.1f}%")

    st.divider()
    st.subheader("Feature Dictionary")
    dict_rows = [
        {
            "Column": r["name"],
            "Display Name": display_name(r["name"]),
            "Type": feature_group(r["name"]),
            "SQLite Type": r["type"],
            "Description": FEATURE_DESCRIPTIONS.get(r["name"], ""),
        }
        for _, r in columns_df.iterrows()
    ]
    st.dataframe(pd.DataFrame(dict_rows), width="stretch", hide_index=True, height=350)

# ---------------------------------------------------------------- Feature Analysis
with tab_feature:
    st.caption("Pick one feature to see its distribution and its relationship with damage grade.")

    feature = st.selectbox("Analyze a feature", ANALYZABLE_FEATURES, format_func=display_name, key="explorer_feature")
    st.caption(f"Type: {feature_group(feature)}")

    is_categorical = feature in CATEGORICAL_FEATURES
    is_numeric = feature in NUMERIC_FEATURES

    st.subheader(f"Distribution: {display_name(feature)}")
    counts = q.feature_value_counts(feature)
    counts_display = counts.copy()
    if is_categorical:
        counts_display["label"] = counts_display["value"].apply(lambda v: code_to_label(feature, v))
        counts_display["value"] = counts_display["value"] + " - " + counts_display["label"]
        counts_display = counts_display.drop(columns="label")

    col_chart, col_table = st.columns([2, 1])
    with col_chart:
        if is_numeric:
            st.plotly_chart(numeric_histogram(counts), width="stretch", key="explorer_feature_numeric_hist")
        else:
            st.plotly_chart(value_counts_bar(counts_display), width="stretch", key="explorer_feature_value_counts_bar")
    with col_table:
        table = counts_display.rename(columns={"value": display_name(feature), "buildings": "Buildings", "pct": "%"})
        table["%"] = table["%"].round(1)
        st.dataframe(table, width="stretch", hide_index=True)

    st.divider()
    st.subheader(f"{display_name(feature)} -> Damage Grade")
    if is_numeric:
        raw = q.feature_and_target_raw(feature)
        st.plotly_chart(numeric_box_by_damage(raw, feature), width="stretch", key="explorer_feature_numeric_box")
    else:
        cross = q.feature_vs_damage(feature)
        pivot_counts = cross.pivot(index="value", columns="damage_grade", values="buildings").fillna(0)
        pivot_pct = pivot_counts.div(pivot_counts.sum(axis=1), axis=0) * 100
        if is_categorical:
            pivot_pct.index = [f"{v} - {code_to_label(feature, v)}" for v in pivot_pct.index]
            pivot_counts.index = pivot_pct.index

        tab_pct, tab_counts = st.tabs(["Row %", "Counts"])
        with tab_pct:
            left, right = st.columns([2, 1])
            with left:
                st.plotly_chart(stacked_damage_bar(pivot_pct), width="stretch", key="explorer_feature_stacked_bar")
            with right:
                st.plotly_chart(damage_heatmap(pivot_pct), width="stretch", key="explorer_feature_heatmap")
        with tab_counts:
            st.dataframe(pivot_counts.astype(int), width="stretch")

# ---------------------------------------------------------------- Geography
with tab_geo:
    st.caption(
        "Full Geo Level 1 -> Level 2 -> Level 3 hierarchy. Size = building count, colour = "
        "average damage grade. These are anonymised region codes, not coordinates — there is "
        "no latitude/longitude in this dataset, so no map is shown."
    )
    nodes = geo.full_hierarchy_nodes()
    g1, g2, g3 = st.columns(3)
    g1.metric("Total buildings", f"{total:,}")
    g2.metric("Regions (Geo Level 1)", int(nodes["parent"].eq("").sum()))
    g3.metric("Total nodes in chart", f"{len(nodes):,}")
    st.plotly_chart(geo_sunburst(nodes), width="stretch", height=800, key="explorer_geo_sunburst")

# ---------------------------------------------------------------- Filtered Explorer
with tab_filter:
    st.caption(
        "Filter the dataset by geography and structural features (SQL `WHERE` + `GROUP BY` — "
        "the full table is never pulled into memory for this)."
    )

    f1, f2, f3 = st.columns(3)
    with f1:
        geo1_choice = st.selectbox("Geo Level 1", ["All"] + q.geo1_options(), key="filter_geo1")
        geo2_opts = ["All"] + q.geo2_options_for_geo1(geo1_choice) if geo1_choice != "All" else ["All"]
        geo2_choice = st.selectbox("Geo Level 2", geo2_opts, key="filter_geo2")
    with f2:
        foundation_choice = st.selectbox(
            "Foundation Type",
            ["All"] + list(q.get_distinct_values("foundation_type")),
            format_func=lambda v: v if v == "All" else f"{v} - {code_to_label('foundation_type', v)}",
            key="filter_foundation",
        )
        roof_choice = st.selectbox(
            "Roof Type",
            ["All"] + list(q.get_distinct_values("roof_type")),
            format_func=lambda v: v if v == "All" else f"{v} - {code_to_label('roof_type', v)}",
            key="filter_roof",
        )
    with f3:
        position_choice = st.selectbox(
            "Position",
            ["All"] + list(q.get_distinct_values("position")),
            format_func=lambda v: v if v == "All" else f"{v} - {code_to_label('position', v)}",
            key="filter_position",
        )
        damage_choice = st.selectbox("Damage Grade", ["All", 1, 2, 3], key="filter_damage")

    filters = {
        "geo_level_1_id": geo1_choice,
        "geo_level_2_id": geo2_choice,
        "foundation_type": foundation_choice,
        "roof_type": roof_choice,
        "position": position_choice,
        "damage_grade": damage_choice,
    }

    summary = q.filtered_summary(filters)
    st.metric("Matching buildings", f"{summary['total']:,}", f"{100 * summary['total'] / total:.1f}% of dataset")

    if summary["total"] == 0:
        st.warning("No buildings match this filter combination.")
    else:
        dist = summary["distribution"]
        cols = st.columns(len(dist)) if len(dist) else []
        for col_widget, (_, row) in zip(cols, dist.iterrows()):
            col_widget.metric(f"Grade {int(row['damage_grade'])}", f"{int(row['buildings']):,}", f"{row['pct']:.1f}%")

        st.divider()
        breakdown_feature = st.selectbox(
            "Break down by feature",
            [c for c in ANALYZABLE_FEATURES if c not in filters],
            format_func=display_name,
            key="filter_breakdown_feature",
        )
        cross = q.filtered_feature_vs_damage(breakdown_feature, filters)
        if not cross.empty:
            pivot_counts = cross.pivot(index="value", columns="damage_grade", values="buildings").fillna(0)
            pivot_pct = pivot_counts.div(pivot_counts.sum(axis=1), axis=0) * 100
            if breakdown_feature in CATEGORICAL_FEATURES:
                pivot_pct.index = [f"{v} - {code_to_label(breakdown_feature, v)}" for v in pivot_pct.index]
            st.plotly_chart(stacked_damage_bar(pivot_pct), width="stretch", key="explorer_filtered_stacked_bar")
