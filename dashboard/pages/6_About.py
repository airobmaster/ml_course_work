import streamlit as st

st.title("ℹ️ About")

st.subheader("The problem")
st.markdown(
    """
After the 7.8-magnitude earthquake that hit Nepal on April 25, 2015, surveys were conducted
to assess damage to hundreds of thousands of buildings. This app predicts a building's
**damage grade** (1 = low, 2 = medium, 3 = near-total destruction) from its structural,
geographic, and ownership characteristics — Kathmandu Living Labs' *Richter's Predictor*
dataset, collected with Nepal's Central Bureau of Statistics.

- **260,601** labeled buildings, **38** raw structural/geographic/ownership features
- Data was collected by Kathmandu Living Labs and the Central Bureau of Statistics under
  Nepal's National Planning Commission
- Official evaluation metric: **micro-averaged F1 score**
"""
)

st.divider()
st.subheader("Model")
st.markdown(
    """
The deployed model is a **LightGBM `LGBMClassifier`** trained directly on the 38 raw feature
columns — no target/frequency encoding, no one-hot encoding, no feature engineering. The 3
`geo_level_*_id` columns are passed as raw integers but declared categorical to LightGBM; the
8 low-cardinality string columns (`foundation_type`, `roof_type`, etc.) are cast to pandas
`category` dtype. LightGBM's native categorical-split handling does the rest.

This reproduces the pipeline in `assets/earthquake_damage.ipynb` ("Step 53 — Train the final
LightGBM model") that produced `assets/submission_lightgbm.csv`. Hyperparameters were tuned
manually in that notebook via sequential single-parameter sweeps against validation Micro-F1:

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 870 |
| `learning_rate` | 0.03 |
| `num_leaves` | 15 |
| `min_child_samples` | 200 |
| `colsample_bytree` | 0.8 |
| `subsample` / `subsample_freq` | 0.8 / 1 |
| `max_depth` | -1 (unlimited, capped by `num_leaves`) |

Neither notebook in this project ever saved a trained model artifact to disk, so
`scripts/train_model.py` retrains this exact configuration and saves it for the app to load
(`dashboard/models/damage_model.joblib`). See the **Analysis Notebook** page for the full,
original notebooks this is built from, and **Model Performance** for held-out metrics.
"""
)

st.divider()
st.subheader("A note on the project's two notebooks")
st.markdown(
    """
This project contains two modeling notebooks that reached different conclusions:

- **`earthquake_damage.ipynb`** ran ablation experiments on geo target-encoding, structural
  interaction features, and age banding — none improved validation Micro-F1 over the plain 38
  raw columns, so its final LightGBM model uses no feature engineering at all. This is the
  notebook the deployed model in this app is built from.
- **`earthquake_damage_full_analysis.ipynb`** takes a different path: a custom
  `GeoHierarchicalTargetEncoder`, engineered interaction/ratio features, and a final
  **LightGBM + CatBoost ensemble** rather than LightGBM alone.

Both are included on the Analysis Notebook page for transparency.
"""
)

st.divider()
st.subheader("Limitations")
st.markdown(
    """
- **Not causal.** The What-if Analysis page shows how the *model's* prediction changes when a
  feature changes — it does not estimate the real-world causal effect of changing that
  building characteristic.
- **Ordinal target treated as multi-class.** `damage_grade` (1 < 2 < 3) has a natural order,
  but this model treats it as unordered multi-class classification, matching the competition's
  own Micro-F1 metric.
- **Class imbalance.** Grade 2 makes up ~57% of the training data, Grade 1 only ~10% —
  Micro-F1, not accuracy, drives evaluation for exactly this reason (see Model Performance).
- **Geographic IDs are anonymised codes**, not coordinates — there is no map view, only the
  ID hierarchy (Geo Level 1 → 2 → 3).
- **Unseen geographic IDs.** A `geo_level_2_id`/`geo_level_3_id` combination never seen in
  training data isn't offered in the Predict Damage / What-if dropdowns, since the app only
  presents combinations actually observed in training.
"""
)

st.divider()
st.subheader("Tech stack")
st.markdown(
    """
Streamlit · LightGBM · scikit-learn · SHAP · pandas · SQLite · Plotly
"""
)
