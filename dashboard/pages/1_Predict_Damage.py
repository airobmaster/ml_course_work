import streamlit as st

from src.input_form import render_building_form
from src.plotting import probability_bar, shap_bar
from src.prediction import get_individual_shap_values, safe_predict_damage

st.title("🏢 Predict Damage")
st.caption(
    "Enter a building's characteristics below, then click **Predict Damage** to run it "
    "through the trained LightGBM model."
)

try:
    values = render_building_form(key_prefix="predict")
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

st.divider()
predict_clicked = st.button("Predict Damage", type="primary")

if predict_clicked:
    result, errors = safe_predict_damage(values)

    if errors:
        st.error("Could not generate a prediction:\n\n" + "\n".join(f"- {e}" for e in errors))
    else:
        st.session_state["last_prediction"] = result
        st.session_state["last_prediction_values"] = values

if "last_prediction" in st.session_state:
    result = st.session_state["last_prediction"]
    grade = result["predicted_class"]
    probs = result["probabilities"]

    grade_names = {1: "Low damage", 2: "Medium damage", 3: "Severe damage / destroyed"}
    grade_colors = {1: "green", 2: "orange", 3: "red"}

    st.divider()
    st.subheader("Prediction")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"### Grade {grade}")
        st.markdown(f":{grade_colors[grade]}[**{grade_names[grade]}**]")
        st.metric("Confidence", f"{probs[grade] * 100:.1f}%")
    with c2:
        st.plotly_chart(probability_bar(probs), width="stretch", key="predict_probability_bar")

    with st.expander("Why this prediction? (SHAP explanation for this building)"):
        st.caption(
            "SHAP values explain **this specific prediction**: how much each feature value "
            "pushed the model toward (red) or away from (blue) the predicted grade. This is "
            "different from the *global* feature importance on the Model Performance page, "
            "which ranks features by their average effect across all predictions, not this "
            "one building."
        )
        try:
            shap_df = get_individual_shap_values(result["input_df"])
            st.plotly_chart(shap_bar(shap_df, top_n=15), width="stretch", key="predict_shap_bar")
        except ImportError:
            st.warning("The `shap` package is not installed, so individual-prediction explanations are unavailable.")
        except Exception as e:
            st.warning(f"Could not compute SHAP explanation: {e}")

    with st.expander("Input record used for this prediction"):
        st.dataframe(result["input_df"], width="stretch", hide_index=True)
