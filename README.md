# Nepal Earthquake — Building Damage Prediction

A course-work data science project that predicts the damage grade of buildings affected by the
2015 Gorkha earthquake in Nepal, based on the Kathmandu Living Labs / Central Bureau of
Statistics survey (the "Richter's Predictor" dataset). The project covers the full workflow —
exploratory data analysis, feature engineering, model comparison, and a final tuned model — and
packages the result as an interactive Streamlit dashboard.

## Purpose

After the 2015 earthquake, hundreds of thousands of buildings needed a damage assessment, far
more than manual survey teams could evaluate quickly. This project explores whether a building's
structural, geographic, and ownership characteristics can be used to predict how severely it was
damaged, as a way of practicing:

- Multi-class classification on an imbalanced, real-world (not toy) dataset
- Feature engineering and encoding strategies for high-cardinality categorical and geographic data
- Model comparison and hyperparameter tuning across several algorithm families
- Model explainability (feature importance and SHAP)
- Turning a trained model into a usable, interactive application

The target variable, `damage_grade`, has three ordinal levels: **1 = low damage**, **2 = medium
damage**, **3 = near-total destruction**. Performance is evaluated using the micro-averaged F1
score, which is appropriate for this imbalanced, multi-class problem.

## Overview of the System

The project is organized into two parts:

1. **Analysis & modeling** (`assets/`, `scripts/`) — Jupyter notebooks and scripts covering
   data cleaning, exploratory data analysis, feature engineering, encoding, model selection,
   tuning, and evaluation. The final model is a tuned **LightGBM** multi-class classifier.
2. **Interactive dashboard** (`dashboard/`) — a multi-page **Streamlit** application that lets a
   user explore the dataset, view model performance, and generate live damage-grade predictions
   (including "what-if" scenario comparisons) using the trained model.

### Technologies Used

| Area | Tools |
|---|---|
| Language | Python |
| Data handling | pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine learning | scikit-learn, LightGBM, XGBoost |
| Explainability | SHAP |
| EDA / profiling | ydata-profiling, Sweetviz |
| Dashboard | Streamlit |
| Storage | SQLite (dashboard data store), joblib (model persistence) |
| Notebooks | Jupyter |

### Repository Structure

```
├── assets/                  Notebooks, problem/metric definitions, generated figures
├── datasets/                Raw competition data (train/test values, train labels)
├── outputs/                 Processed datasets and generated submissions
├── scripts/                 Standalone scripts (model training, feature metadata generation)
├── dashboard/                Streamlit application (see dashboard/README.md for details)
└── *.md                       Supporting documentation (EDA summary, code walkthrough, Q&A)
```

Documentation of the analysis and modeling decisions lives alongside the notebooks:
[`EDA_Summary.md`](EDA_Summary.md), [`CODE_DOCUMENTATION.md`](CODE_DOCUMENTATION.md),
[`Categorical_Feature_Codes.md`](Categorical_Feature_Codes.md), and
[`NOTEBOOK_CELL_INDEX.md`](NOTEBOOK_CELL_INDEX.md).

## The Dashboard

The Streamlit app provides:

- **Home** — project overview and navigation
- **Predict Damage** — enter a building's characteristics and get a live damage-grade prediction
- **What-if Analysis** — compare predictions across changed inputs to see how sensitive the
  prediction is to specific features
- **Dataset Explorer** — interactively query and visualize the underlying survey dataset
- **Model Performance** — held-out validation metrics, confusion matrix, and feature importance
- **Analysis Notebook** — a viewer for the full exploratory/modeling notebook
- **About** — background on the dataset, problem, and modeling approach

## How to Run the App

**Prerequisites:** Python 3.11 and the dependencies in [`requirements.txt`](requirements.txt).

```bash
# 1. Create and activate a virtual environment, then install dependencies
pip install -r requirements.txt

# 2. Build the dashboard's data and model artifacts from the raw datasets
python dashboard/scripts/build_db.py          # datasets/*.csv -> dashboard/data/earthquake.db
python scripts/generate_feature_metadata.py   # datasets/*.csv -> dashboard/config/feature_metadata.json
python scripts/train_model.py                 # datasets/*.csv -> dashboard/models/*

# 3. Launch the dashboard
streamlit run dashboard/app.py
```

The app will open in your browser (default: `http://localhost:8501`). Steps 2 only need to be
re-run if the underlying datasets change; see [`dashboard/README.md`](dashboard/README.md) for
further details on the dashboard's internals.
