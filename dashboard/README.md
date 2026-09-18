# Nepal Earthquake — Building Damage Streamlit App

Predicts `damage_grade` (1 = low, 2 = medium, 3 = severe/destroyed) for a building from its
structural, geographic, and ownership characteristics, using a LightGBM model trained on the
[Richter's Predictor](https://www.drivendata.org/competitions/57/nepal-earthquake/) dataset
(260,601 labeled buildings from the 2015 Nepal earthquake survey).

## Structure

```
dashboard/
    app.py                      # entrypoint; defines page navigation
    pages/
        0_Home.py
        1_Predict_Damage.py
        2_What_If_Analysis.py
        3_Dataset_Explorer.py
        4_Model_Performance.py
        5_Notebook_Viewer.py
        6_About.py
    src/
        database.py              # SQLite connection (cached resource)
        queries.py                # all SQL against the `buildings` table
        geo.py                    # geo hierarchy -> sunburst node table
        labels.py                 # categorical code -> label dictionary, display names
        plotting.py                # all Plotly chart builders
        feature_metadata.py        # loads config/feature_metadata.json
        preprocessing.py           # input validation + training-schema DataFrame builder
        prediction.py               # load model, predict, what-if / scenario comparison, SHAP
        input_form.py                # reusable Streamlit widgets for a building record
    models/
        damage_model.joblib          # trained LightGBM bundle (generated, not hand-written)
        model_metrics.json            # held-out validation metrics + feature importance
    config/
        feature_metadata.json          # per-feature dtype/range/categories, computed from data
    data/
        earthquake.db                   # SQLite: train_values + train_labels joined
        notebook_*.html                  # pre-rendered notebook exports for the viewer page
    scripts/
        build_db.py                       # datasets/*.csv -> data/earthquake.db
    .streamlit/config.toml
```

`scripts/train_model.py` and `scripts/generate_feature_metadata.py` live at the repo root
(next to `datasets/`) since they read `datasets/*.csv` directly; their outputs are written into
`dashboard/models/` and `dashboard/config/`.

## Setup

From the repo root, with the project's `.venv` active:

```bash
python dashboard/scripts/build_db.py          # datasets/*.csv -> dashboard/data/earthquake.db
python scripts/generate_feature_metadata.py   # datasets/*.csv -> dashboard/config/feature_metadata.json
python scripts/train_model.py                 # datasets/*.csv -> dashboard/models/*
streamlit run dashboard/app.py
```

The first three steps only need re-running if `datasets/*.csv` changes. `train_model.py`
retrains and saves the model from scratch (~1 minute); no notebook ever saved a model artifact
to disk, so this step is required before the app's Predict/What-if pages will work.

## Model

A bare LightGBM `LGBMClassifier` trained on the 38 raw feature columns (no encoders, no feature
engineering) — the pipeline in `assets/earthquake_damage.ipynb` ("Step 53") that produced
`assets/submission_lightgbm.csv`. See the in-app **About** page for the full rationale and
hyperparameters, and **Model Performance** for held-out metrics (Micro-F1, confusion matrix,
per-class precision/recall, global feature importance).

## Design notes

- **Prediction inputs always come from the UI**, never from `datasets/test_values.csv` — that
  file is the competition's real held-out test set and is not read anywhere in this app.
- **Dataset Explorer** queries SQLite with `GROUP BY`/aggregation rather than loading the full
  260K-row table into memory for each interaction.
- **Categorical values, numeric ranges, and the geo hierarchy** shown in every dropdown are
  computed directly from `datasets/train_values.csv` (see `scripts/generate_feature_metadata.py`
  and `src/queries.py`'s cascading geo functions) — nothing is hand-invented.
- **What-if Analysis** results are explicitly labeled as model predictions under changed inputs,
  not causal effects.
