import json
from pathlib import Path

import streamlit as st

from src import queries as q

st.title("🏠 Nepal Earthquake — Building Damage Prediction")
st.caption(
    "Predicting `damage_grade` (1 = low, 2 = medium, 3 = severe/destroyed) for buildings "
    "from Kathmandu Living Labs' *Richter's Predictor* dataset, based on the 2015 Gorkha "
    "earthquake survey."
)

metrics_path = Path(__file__).resolve().parent.parent / "models" / "model_metrics.json"

try:
    total_buildings = q.get_total_count()
except FileNotFoundError as e:
    st.error(str(e))
    total_buildings = None

col1, col2, col3, col4 = st.columns(4)
col1.metric("Labeled buildings", f"{total_buildings:,}" if total_buildings else "—")
col2.metric("Raw input features", "38")
col3.metric("Damage grade classes", "3")
col4.metric("Model", "LightGBM (LGBMClassifier)")

if metrics_path.exists():
    with open(metrics_path) as f:
        metrics = json.load(f)
    val = metrics["validation"]
    st.divider()
    st.subheader("Model at a glance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Micro-F1 (held-out)", f"{val['f1_micro']:.4f}", help="The competition's own evaluation metric.")
    m2.metric("Accuracy (held-out)", f"{val['accuracy']:.4f}")
    m3.metric("Macro-F1 (held-out)", f"{val['f1_macro']:.4f}")
    m4.metric("Weighted-F1 (held-out)", f"{val['f1_weighted']:.4f}")
    st.caption(
        "Held-out metrics come from an 80/20 stratified validation split using the same "
        "hyperparameters as the deployed model. See the **Model Performance** page for the "
        "full breakdown."
    )
else:
    st.warning(
        "No trained model artifact found yet. Run `python scripts/train_model.py` from the "
        "repo root to train and save the LightGBM model before using Predict Damage or "
        "What-if Analysis."
    )

st.divider()

st.subheader("What you can do here")
c1, c2 = st.columns(2)
with c1:
    st.markdown(
        """
- **🏢 Predict Damage** — enter a single building's characteristics and get a predicted
  damage grade with class probabilities.
- **🔬 What-if Analysis** — start from a baseline building, change one or more features,
  and see how the model's prediction shifts.
- **📊 Dataset Explorer** — browse the 260,601 labeled buildings: feature distributions,
  damage-grade relationships, and geographic breakdowns, backed by SQLite.
"""
    )
with c2:
    st.markdown(
        """
- **📈 Model Performance** — Micro-F1 (the competition metric), accuracy, per-class
  precision/recall/F1, confusion matrix, and global feature importance.
- **📓 Analysis Notebook** — the full EDA-to-modeling notebooks this app is built on.
- **ℹ️ About** — data source, methodology, and known limitations.
"""
    )

st.divider()
st.info(
    "**Important:** predictions on the *Predict Damage* and *What-if Analysis* pages are "
    "always built from user-entered building characteristics via the UI — never from the "
    "held-out test set. The *Dataset Explorer* page uses the labeled training data purely "
    "for historical analysis, not as a source of prediction inputs.",
    icon="ℹ️",
)
