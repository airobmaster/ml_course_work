from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

st.title("📓 Analysis Notebook")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

NOTEBOOKS = {
    "Full analysis (EDA -> feature engineering -> LightGBM + CatBoost ensemble)": {
        "html": DATA_DIR / "notebook_full_analysis.html",
        "source": "assets/earthquake_damage_full_analysis.ipynb",
        "note": (
            "This notebook's own final section trains a **LightGBM + CatBoost ensemble** "
            "(`submission_ensemble.csv`) — it is not the exact pipeline behind the model "
            "deployed in this app. See the second notebook below for that."
        ),
    },
    "Deployed model source (raw-feature LightGBM, 'Step 53')": {
        "html": DATA_DIR / "notebook_earthquake_damage.html",
        "source": "assets/earthquake_damage.ipynb",
        "note": (
            "The **exact** model shipped in this app (Predict Damage, What-if Analysis) is "
            "reproduced from this notebook's 'Step 53 — Train the final LightGBM model' "
            "section: a bare `LGBMClassifier` on the 38 raw feature columns, no encoders, "
            "which produced `assets/submission_lightgbm.csv`."
        ),
    },
}

choice = st.selectbox("Notebook", list(NOTEBOOKS.keys()))
info = NOTEBOOKS[choice]

st.caption(f"Source file: `{info['source']}`")
st.info(info["note"], icon="ℹ️")

if not info["html"].exists():
    st.error(
        f"Rendered notebook not found at {info['html']}. Regenerate it with:\n\n"
        f"`jupyter nbconvert --to html --output-dir dashboard/data {info['source']}`"
    )
else:
    html = info["html"].read_text(encoding="utf-8")
    st.caption(f"Rendered from the notebook's saved outputs ({info['html'].stat().st_size / 1e6:.1f} MB). Scroll within the frame below.")
    components.html(html, height=1000, scrolling=True)
