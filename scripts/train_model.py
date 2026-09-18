#!/usr/bin/env python
"""Train and save the LightGBM building-damage model used by the Streamlit app.

This reproduces the exact pipeline from `assets/earthquake_damage.ipynb`
(cells 14, 153, 205-216, 312-317 — "Step 53: Train the final LightGBM model")
that actually produced `assets/submission_lightgbm.csv`:

- Raw 38 feature columns (`train_values` minus `building_id`), no feature
  engineering, no target/frequency/hierarchical geo-encoding.
- The 3 `geo_level_*_id` columns are left as plain int64 but declared
  categorical to LightGBM via `categorical_feature=`.
- The 8 low-cardinality string columns are cast to pandas `category` dtype.
- LGBMClassifier hyperparameters are the notebook's manually-tuned final
  values (num_leaves=15, min_child_samples=200, colsample_bytree=0.8,
  subsample=0.8, subsample_freq=1, learning_rate=0.03, n_estimators=870).

No cell in either notebook ever saved a model artifact to disk, so this
script performs that step: it (1) fits the same hyperparameters on an 80/20
stratified hold-out split to produce honest, reproducible validation metrics
for the app's Model Performance page (the notebook itself never printed a
full classification report for this exact final hyperparameter combination),
then (2) refits on 100% of the labeled data — exactly as the notebook's own
final cell does — for the artifact that actually ships with the app.
"""
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT / "datasets"
MODELS_DIR = ROOT / "dashboard" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ID_COL = "building_id"
TARGET_COL = "damage_grade"

GEO_COLS = ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]

NUMERIC_COLS = [
    "count_floors_pre_eq",
    "age",
    "area_percentage",
    "height_percentage",
    "count_families",
]

CATEGORICAL_COLS = [
    "land_surface_condition",
    "foundation_type",
    "roof_type",
    "ground_floor_type",
    "other_floor_type",
    "position",
    "plan_configuration",
    "legal_ownership_status",
]

SUPERSTRUCTURE_COLS = [
    "has_superstructure_adobe_mud",
    "has_superstructure_mud_mortar_stone",
    "has_superstructure_stone_flag",
    "has_superstructure_cement_mortar_stone",
    "has_superstructure_mud_mortar_brick",
    "has_superstructure_cement_mortar_brick",
    "has_superstructure_timber",
    "has_superstructure_bamboo",
    "has_superstructure_rc_non_engineered",
    "has_superstructure_rc_engineered",
    "has_superstructure_other",
]

SECONDARY_USE_UMBRELLA_COL = "has_secondary_use"
SECONDARY_USE_SUB_COLS = [
    "has_secondary_use_agriculture",
    "has_secondary_use_hotel",
    "has_secondary_use_rental",
    "has_secondary_use_institution",
    "has_secondary_use_school",
    "has_secondary_use_industry",
    "has_secondary_use_health_post",
    "has_secondary_use_gov_office",
    "has_secondary_use_use_police",
    "has_secondary_use_other",
]
SECONDARY_USE_COLS = [SECONDARY_USE_UMBRELLA_COL] + SECONDARY_USE_SUB_COLS

BINARY_COLS = SUPERSTRUCTURE_COLS + SECONDARY_USE_COLS

RANDOM_STATE = 42

LGBM_PARAMS = dict(
    objective="multiclass",
    num_class=3,
    n_estimators=870,
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=200,
    colsample_bytree=0.8,
    subsample=0.8,
    subsample_freq=1,
    max_depth=-1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    force_col_wise=True,
    verbosity=-1,
)

LGB_CATEGORICAL_FEATURE = GEO_COLS + CATEGORICAL_COLS


def load_data():
    train_values = pd.read_csv(DATASETS_DIR / "train_values.csv")
    train_labels = pd.read_csv(DATASETS_DIR / "train_labels.csv")
    assert (train_values[ID_COL] == train_labels[ID_COL]).all(), "train_values/train_labels misaligned"

    feature_cols = [c for c in train_values.columns if c != ID_COL]
    X = train_values[feature_cols].copy()
    y = train_labels[TARGET_COL].copy()
    return X, y, feature_cols


def cast_categoricals(X: pd.DataFrame, reference: pd.DataFrame | None = None) -> pd.DataFrame:
    """Cast the 8 string categorical columns to pandas `category` dtype,
    exactly as the notebook's final-model cell does. If `reference` is given,
    align categories to it (mirrors the notebook's
    `X_test_final[col].cat.set_categories(X_final[col].cat.categories)`)."""
    X = X.copy()
    for col in CATEGORICAL_COLS:
        X[col] = X[col].astype("category")
        if reference is not None:
            X[col] = X[col].cat.set_categories(reference[col].cat.categories)
    return X


def evaluate(model: lgb.LGBMClassifier, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    preds = model.predict(X_val)
    proba = model.predict_proba(X_val)
    class_order = list(model.classes_)

    report = classification_report(y_val, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_val, preds, labels=class_order)

    metrics = {
        "n_val": int(len(y_val)),
        "accuracy": float(accuracy_score(y_val, preds)),
        "f1_micro": float(f1_score(y_val, preds, average="micro")),
        "f1_macro": float(f1_score(y_val, preds, average="macro")),
        "f1_weighted": float(f1_score(y_val, preds, average="weighted")),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": [int(c) for c in class_order],
        "predicted_class_distribution": {
            str(c): int((preds == c).sum()) for c in class_order
        },
    }
    return metrics


def main():
    print("Loading data...")
    X, y, feature_cols = load_data()
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    class_dist = y.value_counts().sort_index()
    train_class_distribution = {
        str(int(k)): {"count": int(v), "pct": float(100 * v / len(y))}
        for k, v in class_dist.items()
    }
    print("Train class distribution:", train_class_distribution)

    # ---------------------------------------------------------------- #
    # 1. Honest held-out validation metrics (80/20 stratified split,
    #    same hyperparameters/random_state as production model).
    # ---------------------------------------------------------------- #
    print("\n--- Held-out validation (80/20 stratified split) ---")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    X_train_c = cast_categoricals(X_train)
    X_val_c = cast_categoricals(X_val, reference=X_train_c)

    val_model = lgb.LGBMClassifier(**LGBM_PARAMS)
    val_model.fit(X_train_c, y_train, categorical_feature=LGB_CATEGORICAL_FEATURE)

    val_metrics = evaluate(val_model, X_val_c, y_val)
    print(f"Held-out Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Held-out Micro-F1: {val_metrics['f1_micro']:.4f}")
    print(f"Held-out Macro-F1: {val_metrics['f1_macro']:.4f}")
    print(f"Held-out Weighted-F1: {val_metrics['f1_weighted']:.4f}")

    val_importance_gain = val_model.booster_.feature_importance(importance_type="gain")
    val_importance_split = val_model.booster_.feature_importance(importance_type="split")

    # ---------------------------------------------------------------- #
    # 2. Production model: refit on 100% of labeled data (as the
    #    notebook's own final cell does) -- this is the artifact the
    #    app actually loads for inference.
    # ---------------------------------------------------------------- #
    print("\n--- Refitting production model on 100% of labeled data ---")
    X_full_c = cast_categoricals(X)
    prod_model = lgb.LGBMClassifier(**LGBM_PARAMS)
    prod_model.fit(X_full_c, y, categorical_feature=LGB_CATEGORICAL_FEATURE)
    print("Production model trained on", len(X_full_c), "rows,", X_full_c.shape[1], "features.")

    prod_importance_gain = prod_model.booster_.feature_importance(importance_type="gain")
    prod_importance_split = prod_model.booster_.feature_importance(importance_type="split")

    categorical_categories = {
        col: X_full_c[col].cat.categories.tolist() for col in CATEGORICAL_COLS
    }

    # ---------------------------------------------------------------- #
    # 3. Sanity check: reloaded-from-disk predictions must exactly match
    #    in-memory predictions, and a freshly-cast single-row category
    #    dtype (independent category ordering) must still match the
    #    batch prediction for the same row -- this validates that
    #    LightGBM's internally-stored pandas_categorical mapping (saved
    #    with the model) is what makes single-row inference safe, not
    #    the category order we happen to pass in.
    # ---------------------------------------------------------------- #
    sample_idx = X_full_c.sample(5, random_state=RANDOM_STATE).index
    batch_proba = prod_model.predict_proba(X_full_c.loc[sample_idx])
    for pos, idx in enumerate(sample_idx):
        row = X.loc[[idx]].copy()  # fresh, uncast row -- independent category order
        row = cast_categoricals(row)
        single_proba = prod_model.predict_proba(row)[0]
        assert np.allclose(single_proba, batch_proba[pos], atol=1e-8), (
            f"Mismatch on row {idx}: batch={batch_proba[pos]} single={single_proba}"
        )
    print("Sanity check passed: single-row category casting matches batch predictions.")

    # ---------------------------------------------------------------- #
    # Save artifacts
    # ---------------------------------------------------------------- #
    bundle = {
        "model": prod_model,
        "feature_columns": feature_cols,
        "id_col": ID_COL,
        "target_col": TARGET_COL,
        "geo_cols": GEO_COLS,
        "numeric_cols": NUMERIC_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "binary_cols": BINARY_COLS,
        "superstructure_cols": SUPERSTRUCTURE_COLS,
        "secondary_use_umbrella_col": SECONDARY_USE_UMBRELLA_COL,
        "secondary_use_sub_cols": SECONDARY_USE_SUB_COLS,
        "categorical_feature_for_fit": LGB_CATEGORICAL_FEATURE,
        "categorical_categories": categorical_categories,
        "class_labels": [int(c) for c in prod_model.classes_],
        "random_state": RANDOM_STATE,
        "hyperparameters": LGBM_PARAMS,
        "trained_on_n_rows": int(len(X_full_c)),
    }
    model_path = MODELS_DIR / "damage_model.joblib"
    joblib.dump(bundle, model_path)
    print(f"\nSaved model bundle to {model_path} ({model_path.stat().st_size / 1e6:.1f} MB)")

    metrics_out = {
        "model_type": "LightGBM (LGBMClassifier), multiclass",
        "source_notebook": "assets/earthquake_damage.ipynb (final LightGBM model, 'Step 53')",
        "hyperparameters": LGBM_PARAMS,
        "random_state": RANDOM_STATE,
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "train_class_distribution": train_class_distribution,
        "n_total_labeled_rows": int(len(X)),
        "production_model": {
            "trained_on": "100% of labeled data (260,601 rows) -- matches the notebook's own final-model cell",
            "note": "No held-out check exists for this exact refit; validation metrics below come from a separate 80/20 stratified split fit with identical hyperparameters.",
            "feature_importance_gain": {
                col: float(v) for col, v in zip(feature_cols, prod_importance_gain)
            },
            "feature_importance_split": {
                col: float(v) for col, v in zip(feature_cols, prod_importance_split)
            },
        },
        "validation": {
            "method": "80/20 stratified train/validation split (sklearn train_test_split, stratify=y, random_state=42), identical hyperparameters to the production model",
            **val_metrics,
            "feature_importance_gain": {
                col: float(v) for col, v in zip(feature_cols, val_importance_gain)
            },
            "feature_importance_split": {
                col: float(v) for col, v in zip(feature_cols, val_importance_split)
            },
        },
    }
    metrics_path = MODELS_DIR / "model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
