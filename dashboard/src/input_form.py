"""Reusable building-input form used by both the Predict Damage and What-if
Analysis pages. Widget choice follows the actual feature type from
config/feature_metadata.json: number_input/slider for numeric columns,
selectbox for categorical columns (labelled with the real category meaning),
cascading selectboxes for the 3 geo levels (options restricted to
combinations actually observed in training data), and checkboxes for binary
flags.
"""

import streamlit as st

from src import queries as q
from src.feature_metadata import load_feature_metadata


def render_geo_inputs(key_prefix: str, defaults: dict | None = None) -> dict:
    defaults = defaults or {}
    meta = load_feature_metadata()

    geo1_opts = q.geo1_options()
    geo1_default = defaults.get("geo_level_1_id")
    geo1_index = geo1_opts.index(geo1_default) if geo1_default in geo1_opts else 0
    geo1 = st.selectbox(
        meta["features"]["geo_level_1_id"]["display_name"],
        geo1_opts,
        index=geo1_index,
        key=f"{key_prefix}_geo1",
        help=meta["features"]["geo_level_1_id"]["description"],
    )

    geo2_opts = q.geo2_options_for_geo1(geo1)
    geo2_default = defaults.get("geo_level_2_id")
    geo2_index = geo2_opts.index(geo2_default) if geo2_default in geo2_opts else 0
    geo2 = st.selectbox(
        meta["features"]["geo_level_2_id"]["display_name"],
        geo2_opts,
        index=geo2_index,
        key=f"{key_prefix}_geo2",
        help="Options are restricted to geo_level_2_id values actually observed under the selected geo_level_1_id in training data.",
    )

    geo3_opts = q.geo3_options_for_geo2(geo1, geo2)
    geo3_default = defaults.get("geo_level_3_id")
    geo3_index = geo3_opts.index(geo3_default) if geo3_default in geo3_opts else 0
    geo3 = st.selectbox(
        meta["features"]["geo_level_3_id"]["display_name"],
        geo3_opts,
        index=geo3_index,
        key=f"{key_prefix}_geo3",
        help="Options are restricted to geo_level_3_id values actually observed under the selected geo_level_1/2_id in training data.",
    )

    return {"geo_level_1_id": geo1, "geo_level_2_id": geo2, "geo_level_3_id": geo3}


def render_numeric_inputs(key_prefix: str, defaults: dict | None = None) -> dict:
    defaults = defaults or {}
    meta = load_feature_metadata()
    values = {}
    cols = st.columns(len(meta["numeric_cols"]))
    for col_widget, feature in zip(cols, meta["numeric_cols"]):
        fmeta = meta["features"][feature]
        default_val = defaults.get(feature, round(fmeta["median"]))
        with col_widget:
            values[feature] = st.number_input(
                fmeta["display_name"],
                min_value=int(fmeta["min"]),
                max_value=int(fmeta["max"]),
                value=int(default_val),
                key=f"{key_prefix}_{feature}",
                help=f"{fmeta['description']} Training data range: {fmeta['min']}-{fmeta['max']} (median {fmeta['median']:.0f}).",
            )
    return values


def render_categorical_inputs(key_prefix: str, defaults: dict | None = None) -> dict:
    defaults = defaults or {}
    meta = load_feature_metadata()
    values = {}
    cols = st.columns(2)
    for i, feature in enumerate(meta["categorical_cols"]):
        fmeta = meta["features"][feature]
        codes = [v["code"] for v in fmeta["values"]]
        code_to_label = {v["code"]: v["label"] for v in fmeta["values"]}
        default_val = defaults.get(feature, fmeta["default"])
        index = codes.index(default_val) if default_val in codes else 0
        with cols[i % 2]:
            values[feature] = st.selectbox(
                fmeta["display_name"],
                codes,
                index=index,
                format_func=lambda c, m=code_to_label: f"{c} - {m[c]}",
                key=f"{key_prefix}_{feature}",
                help=fmeta["description"],
            )
    return values


def render_binary_inputs(key_prefix: str, defaults: dict | None = None) -> dict:
    """Superstructure material flags (multi-select, any combination allowed
    -- 68% of buildings report exactly one, but multi-material buildings are
    real) plus secondary-use sub-flags. `has_secondary_use` is not exposed
    directly: it's derived as the logical OR of the sub-flags, mirroring the
    perfect (0-mismatch) invariant confirmed in the training data."""
    defaults = defaults or {}
    meta = load_feature_metadata()
    values = {}

    st.caption("Superstructure material(s) -- select all that apply.")
    cols = st.columns(3)
    for i, feature in enumerate(meta["superstructure_cols"]):
        fmeta = meta["features"][feature]
        default_val = bool(defaults.get(feature, fmeta["default"]))
        with cols[i % 3]:
            checked = st.checkbox(
                fmeta["display_name"].replace("Has Superstructure ", ""),
                value=default_val,
                key=f"{key_prefix}_{feature}",
                help=fmeta["description"],
            )
        values[feature] = int(checked)

    st.caption("Secondary use(s) -- select all that apply.")
    cols = st.columns(3)
    for i, feature in enumerate(meta["secondary_use_sub_cols"]):
        fmeta = meta["features"][feature]
        default_val = bool(defaults.get(feature, fmeta["default"]))
        with cols[i % 3]:
            checked = st.checkbox(
                fmeta["display_name"].replace("Has Secondary Use ", ""),
                value=default_val,
                key=f"{key_prefix}_{feature}",
                help=fmeta["description"],
            )
        values[feature] = int(checked)

    values["has_secondary_use"] = int(any(values[c] for c in meta["secondary_use_sub_cols"]))
    return values


def render_feature_widget(feature: str, key: str, default):
    """Render a single appropriately-typed widget (number_input / selectbox
    / checkbox) for one non-geo feature -- used by the What-if page's
    advanced multi-feature scenario builder, where only a user-chosen subset
    of features needs a widget rather than the full form."""
    meta = load_feature_metadata()
    fmeta = meta["features"][feature]

    if fmeta["feature_type"] == "numeric":
        return st.number_input(
            fmeta["display_name"],
            min_value=int(fmeta["min"]),
            max_value=int(fmeta["max"]),
            value=int(default) if default is not None else round(fmeta["median"]),
            key=key,
            help=fmeta["description"],
        )
    if fmeta["feature_type"] == "categorical":
        codes = [v["code"] for v in fmeta["values"]]
        code_to_label = {v["code"]: v["label"] for v in fmeta["values"]}
        index = codes.index(default) if default in codes else 0
        return st.selectbox(
            fmeta["display_name"],
            codes,
            index=index,
            format_func=lambda c, m=code_to_label: f"{c} - {m[c]}",
            key=key,
            help=fmeta["description"],
        )
    if fmeta["feature_type"] == "binary":
        return int(
            st.checkbox(
                fmeta["display_name"],
                value=bool(default),
                key=key,
                help=fmeta["description"],
            )
        )
    raise ValueError(f"render_feature_widget doesn't support feature_type={fmeta['feature_type']!r}")


def render_building_form(key_prefix: str, defaults: dict | None = None) -> dict:
    """Render the full building-input form (geo + numeric + categorical +
    binary flags) and return the current widget values as a flat dict, in
    no particular order. `key_prefix` must be unique per form instance on
    the page (so Predict and What-if forms don't collide in session_state)."""
    values = {}

    st.markdown("**Location**")
    values.update(render_geo_inputs(key_prefix, defaults))

    st.markdown("**Structural characteristics**")
    values.update(render_numeric_inputs(key_prefix, defaults))
    values.update(render_categorical_inputs(key_prefix, defaults))

    with st.expander("Superstructure material & secondary use", expanded=False):
        values.update(render_binary_inputs(key_prefix, defaults))

    return values
