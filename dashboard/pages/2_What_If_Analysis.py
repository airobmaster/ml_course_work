import streamlit as st

from src.feature_metadata import load_feature_metadata
from src.input_form import render_building_form, render_feature_widget
from src.plotting import probability_bar, scenario_comparison_chart, whatif_probability_chart
from src.prediction import ValidationError, compare_scenarios, safe_predict_damage, what_if_single_feature

st.title("🔬 What-if Analysis")
st.caption(
    "These are **model what-if predictions**, not causal effects — they show how the "
    "trained LightGBM model's output changes when a feature value changes, holding "
    "everything else fixed. They do not claim that changing the real building would cause "
    "the same change in real-world damage."
)

st.divider()
st.subheader("1. Build a baseline building")

try:
    baseline_form_values = render_building_form(key_prefix="whatif")
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

if st.button("Set as Baseline", type="primary"):
    result, errors = safe_predict_damage(baseline_form_values)
    if errors:
        st.error("Could not compute a baseline prediction:\n\n" + "\n".join(f"- {e}" for e in errors))
    else:
        st.session_state["whatif_baseline_values"] = baseline_form_values
        st.session_state["whatif_baseline_result"] = result
        st.session_state["whatif_scenarios"] = {}  # reset saved scenarios when baseline changes

if "whatif_baseline_result" not in st.session_state:
    st.info("Set a baseline above to unlock the what-if tools below.")
    st.stop()

baseline_values = st.session_state["whatif_baseline_values"]
baseline_result = st.session_state["whatif_baseline_result"]

st.divider()
st.subheader("Baseline prediction")
bc1, bc2 = st.columns([1, 2])
with bc1:
    st.metric("Predicted grade", baseline_result["predicted_class"])
    st.metric("Confidence", f"{baseline_result['probabilities'][baseline_result['predicted_class']] * 100:.1f}%")
with bc2:
    st.plotly_chart(probability_bar(baseline_result["probabilities"]), width="stretch", key="whatif_baseline_probability_bar")

st.divider()
tab_single, tab_advanced = st.tabs(["Single-feature what-if", "Advanced: multi-feature scenarios"])

meta = load_feature_metadata()
scannable_features = meta["categorical_cols"] + meta["numeric_cols"] + [
    c for c in meta["superstructure_cols"] + meta["secondary_use_sub_cols"]
]

with tab_single:
    st.caption(
        "Pick one feature. Every other field stays at the baseline value while this one is "
        "swept across its possible values."
    )
    feature = st.selectbox(
        "Feature to vary",
        scannable_features,
        format_func=lambda c: meta["features"][c]["display_name"],
        key="whatif_single_feature",
    )
    try:
        scan_df = what_if_single_feature(baseline_values, feature)
    except ValueError as e:
        st.warning(str(e))
        scan_df = None

    if scan_df is not None and not scan_df.empty:
        st.plotly_chart(
            whatif_probability_chart(scan_df, meta["features"][feature]["display_name"]),
            width="stretch",
            key="whatif_single_feature_chart",
        )
        display_df = scan_df.rename(
            columns={"label": meta["features"][feature]["display_name"], "predicted_grade": "Predicted Grade"}
        ).drop(columns=["value", "is_baseline"])
        st.dataframe(display_df, width="stretch", hide_index=True)
        st.caption(
            f"Baseline value: **{baseline_values.get(feature)}**. Rows above show the predicted "
            "grade and class probabilities if that one field were changed to each listed value."
        )

with tab_advanced:
    st.caption(
        "Choose several features to change at once, set their new values, then add the "
        "scenario to compare it against the baseline (and against other saved scenarios)."
    )
    chosen = st.multiselect(
        "Features to change",
        scannable_features,
        format_func=lambda c: meta["features"][c]["display_name"],
        key="whatif_adv_features",
    )

    overrides = {}
    if chosen:
        cols = st.columns(min(3, len(chosen)))
        for i, feat in enumerate(chosen):
            with cols[i % len(cols)]:
                overrides[feat] = render_feature_widget(
                    feat, key=f"whatif_adv_widget_{feat}", default=baseline_values.get(feat)
                )

    scenario_name = st.text_input(
        "Scenario name", value=f"Scenario {len(st.session_state.get('whatif_scenarios', {})) + 1}"
    )
    add_col, clear_col = st.columns([1, 1])
    with add_col:
        if st.button("Add scenario", disabled=not chosen):
            st.session_state.setdefault("whatif_scenarios", {})
            st.session_state["whatif_scenarios"][scenario_name] = overrides
    with clear_col:
        if st.button("Clear all scenarios"):
            st.session_state["whatif_scenarios"] = {}

    scenarios = st.session_state.get("whatif_scenarios", {})
    if scenarios:
        try:
            comparison_df = compare_scenarios(baseline_values, scenarios)
        except ValidationError as e:
            st.error("A saved scenario is no longer valid:\n\n" + "\n".join(f"- {err}" for err in e.errors))
        else:
            st.plotly_chart(scenario_comparison_chart(comparison_df), width="stretch", key="whatif_scenario_comparison_chart")
            st.dataframe(
                comparison_df.rename(columns={"scenario": "Scenario", "predicted_grade": "Predicted Grade"}),
                width="stretch",
                hide_index=True,
            )
            with st.expander("Scenario definitions (feature overrides vs. baseline)"):
                for name, ov in scenarios.items():
                    st.markdown(f"**{name}**")
                    st.json({meta['features'][k]['display_name']: v for k, v in ov.items()})
    else:
        st.info("Add at least one scenario above to see the comparison.")
