"""Reusable inference logic: load the saved LightGBM bundle once, predict a
single building's damage grade with class probabilities, and run
what-if / multi-scenario comparisons by varying a baseline record.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.feature_metadata import load_feature_metadata
from src.preprocessing import ValidationError, build_input_dataframe

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "damage_model.joblib"


@st.cache_resource
def load_model_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. Run "
            "`python scripts/train_model.py` from the repo root first."
        )
    return joblib.load(MODEL_PATH)


@st.cache_resource
def get_shap_explainer():
    """A shap.TreeExplainer is cheap to build once (it just wraps the
    booster) and fast per-row to evaluate -- built lazily since shap isn't
    needed on every page."""
    import shap

    bundle = load_model_bundle()
    return shap.TreeExplainer(bundle["model"].booster_)


def predict_damage(values: dict) -> dict:
    """Validate `values` (a raw {feature: value} building record), run it
    through the saved pipeline, and return the predicted grade + class
    probabilities. Raises ValidationError on bad input."""
    bundle = load_model_bundle()
    model = bundle["model"]

    df = build_input_dataframe(values)  # raises ValidationError
    proba = model.predict_proba(df)[0]
    class_labels = list(model.classes_)
    probabilities = {int(c): float(p) for c, p in zip(class_labels, proba)}
    predicted_class = int(class_labels[proba.argmax()])

    return {
        "predicted_class": predicted_class,
        "probabilities": probabilities,
        "input_df": df,
    }


def safe_predict_damage(values: dict) -> tuple[dict | None, list[str]]:
    """Same as predict_damage but never raises -- returns (result, errors).
    Intended for Streamlit callers that want to render errors inline rather
    than catch exceptions."""
    try:
        return predict_damage(values), []
    except ValidationError as e:
        return None, e.errors
    except Exception as e:  # model/file errors, etc.
        return None, [str(e)]


def what_if_single_feature(baseline_values: dict, feature: str) -> pd.DataFrame:
    """Vary one feature across every valid value it can take (from the
    training-data-derived metadata), holding all other fields at the
    baseline, and return a comparison table: value, predicted grade,
    P(grade=1..3). This is a model *what-if* scan, not a causal estimate."""
    meta = load_feature_metadata()
    fmeta = meta["features"][feature]

    if fmeta["feature_type"] == "categorical":
        candidates = [v["code"] for v in fmeta["values"]]
    elif fmeta["feature_type"] == "binary":
        candidates = [0, 1]
    elif fmeta["feature_type"] == "numeric":
        # Representative values from the training-data distribution (not
        # invented): min, 1st percentile, median, 99th percentile, max, plus
        # the baseline's own current value so it's always shown.
        raw = {
            int(round(fmeta["min"])),
            int(round(fmeta["p1"])),
            int(round(fmeta["median"])),
            int(round(fmeta["p99"])),
            int(round(fmeta["max"])),
            int(baseline_values.get(feature, fmeta["median"])),
        }
        candidates = sorted(raw)
    elif fmeta["feature_type"] == "geo":
        raise ValueError(
            f"{fmeta['display_name']} is a geographic identifier -- use the location fields "
            "in the baseline form instead of the single-feature scan."
        )
    else:
        raise ValueError(f"Unsupported feature type for what-if scan: {fmeta['feature_type']}")

    rows = []
    for candidate in candidates:
        scenario = dict(baseline_values)
        scenario[feature] = candidate
        result, errors = safe_predict_damage(scenario)
        if errors:
            continue
        label = candidate
        if fmeta["feature_type"] == "categorical":
            label = next(v["label"] for v in fmeta["values"] if v["code"] == candidate)
        rows.append(
            {
                "value": candidate,
                "label": label,
                "predicted_grade": result["predicted_class"],
                "P(Grade 1)": result["probabilities"].get(1, 0.0),
                "P(Grade 2)": result["probabilities"].get(2, 0.0),
                "P(Grade 3)": result["probabilities"].get(3, 0.0),
                "is_baseline": candidate == baseline_values.get(feature),
            }
        )
    return pd.DataFrame(rows)


def compare_scenarios(baseline_values: dict, scenario_values: dict) -> pd.DataFrame:
    """Compare the baseline record against one or more named scenarios
    (each a full or partial override dict merged onto the baseline).
    `scenario_values` maps scenario name -> override dict."""
    rows = []
    baseline_result, errors = safe_predict_damage(baseline_values)
    if errors:
        raise ValidationError(errors)
    rows.append(
        {
            "scenario": "Baseline",
            "predicted_grade": baseline_result["predicted_class"],
            "P(Grade 1)": baseline_result["probabilities"].get(1, 0.0),
            "P(Grade 2)": baseline_result["probabilities"].get(2, 0.0),
            "P(Grade 3)": baseline_result["probabilities"].get(3, 0.0),
        }
    )
    for name, overrides in scenario_values.items():
        merged = dict(baseline_values)
        merged.update(overrides)
        result, errors = safe_predict_damage(merged)
        if errors:
            raise ValidationError(errors)
        rows.append(
            {
                "scenario": name,
                "predicted_grade": result["predicted_class"],
                "P(Grade 1)": result["probabilities"].get(1, 0.0),
                "P(Grade 2)": result["probabilities"].get(2, 0.0),
                "P(Grade 3)": result["probabilities"].get(3, 0.0),
            }
        )
    return pd.DataFrame(rows)


def get_global_feature_importance(importance_type: str = "gain") -> pd.DataFrame:
    """LightGBM's built-in feature importance for the production model --
    a *global* ranking (which features matter most across all predictions),
    not specific to any single building. Distinct from SHAP, which explains
    one prediction at a time (see get_individual_shap_values)."""
    bundle = load_model_bundle()
    model = bundle["model"]
    importances = model.booster_.feature_importance(importance_type=importance_type)
    df = pd.DataFrame({"feature": bundle["feature_columns"], "importance": importances})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    total = df["importance"].sum()
    df["importance_pct"] = 100 * df["importance"] / total if total > 0 else 0.0
    return df


def get_individual_shap_values(input_df: pd.DataFrame) -> pd.DataFrame:
    """SHAP values for one specific prediction -- explains *this building's*
    predicted class in terms of how much each feature value pushed the
    prediction toward or away from it, unlike the global importance above."""
    bundle = load_model_bundle()
    model = bundle["model"]
    explainer = get_shap_explainer()

    predicted_class = int(model.predict(input_df)[0])
    class_index = list(model.classes_).index(predicted_class)

    shap_values = explainer.shap_values(input_df)
    # shap.TreeExplainer on a multiclass LightGBM booster returns either a
    # list of per-class arrays or a single (n_rows, n_features, n_classes)
    # array depending on shap version -- handle both.
    if isinstance(shap_values, list):
        row_values = shap_values[class_index][0]
    elif shap_values.ndim == 3:
        row_values = shap_values[0, :, class_index]
    else:
        row_values = shap_values[0]

    df = pd.DataFrame(
        {
            "feature": input_df.columns,
            "value": input_df.iloc[0].astype(str).values,
            "shap_value": row_values,
        }
    )
    df["abs_shap"] = df["shap_value"].abs()
    df = df.sort_values("abs_shap", ascending=False).reset_index(drop=True)
    df["explained_class"] = predicted_class
    return df
