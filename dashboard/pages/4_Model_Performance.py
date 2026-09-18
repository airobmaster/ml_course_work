import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.labels import display_name
from src.plotting import confusion_matrix_heatmap, feature_importance_bar

st.title("📈 Model Performance")

METRICS_PATH = Path(__file__).resolve().parent.parent / "models" / "model_metrics.json"

if not METRICS_PATH.exists():
    st.error(
        f"No metrics file found at {METRICS_PATH}. Run `python scripts/train_model.py` "
        "from the repo root to train the model and generate metrics."
    )
    st.stop()

with open(METRICS_PATH) as f:
    metrics = json.load(f)

val = metrics["validation"]

st.caption(
    f"{metrics['model_type']}, trained per {metrics['source_notebook']}. Validation metrics "
    f"below come from an {val['method'].split(',')[0]} using the exact hyperparameters of the "
    "deployed model (the notebook itself never logged a full classification report for this "
    "precise final configuration, so this evaluation was run to produce one honestly)."
)

st.divider()
st.subheader("Headline metrics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Micro-F1 ⭐", f"{val['f1_micro']:.4f}", help="The competition's own evaluation metric — highlighted.")
c2.metric("Accuracy", f"{val['accuracy']:.4f}")
c3.metric("Macro-F1", f"{val['f1_macro']:.4f}")
c4.metric("Weighted-F1", f"{val['f1_weighted']:.4f}")
st.caption(f"Computed on {val['n_val']:,} held-out validation buildings (stratified 80/20 split, random_state={metrics['random_state']}).")

st.divider()
st.subheader("Per-class metrics")
report = val["classification_report"]
class_rows = []
for label in ("1", "2", "3"):
    r = report[label]
    class_rows.append(
        {
            "Grade": f"Grade {label}",
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1-score": r["f1-score"],
            "Support": int(r["support"]),
        }
    )
class_df = pd.DataFrame(class_rows)
st.dataframe(
    class_df.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1-score": "{:.3f}", "Support": "{:,}"}),
    width="stretch",
    hide_index=True,
)
avg1, avg2 = st.columns(2)
avg1.metric("Macro avg F1", f"{report['macro avg']['f1-score']:.3f}")
avg2.metric("Weighted avg F1", f"{report['weighted avg']['f1-score']:.3f}")

st.divider()
st.subheader("Confusion matrix")
cm = val["confusion_matrix"]
labels = val["confusion_matrix_labels"]
st.plotly_chart(confusion_matrix_heatmap(cm, labels), width="stretch", key="perf_confusion_matrix")
st.caption("Rows = actual grade, columns = predicted grade, on the held-out validation split.")

st.divider()
st.subheader("Target class distribution")
st.caption("Class imbalance in the full labeled dataset — the reason Micro-F1 (not accuracy) drives evaluation.")
dist = metrics["train_class_distribution"]
d1, d2, d3 = st.columns(3)
for col_widget, grade in zip((d1, d2, d3), ("1", "2", "3")):
    col_widget.metric(f"Grade {grade}", f"{dist[grade]['count']:,}", f"{dist[grade]['pct']:.1f}%")

st.divider()
st.subheader("Global feature importance")
st.caption(
    "LightGBM's built-in **gain**-based importance, computed on the production model — the "
    "total reduction in training loss each feature contributed across all splits, aggregated "
    "over every prediction. This is a *global* ranking, not specific to any one building "
    "(compare with the per-prediction SHAP explanation on the Predict Damage page)."
)
importance = metrics["production_model"]["feature_importance_gain"]
imp_df = pd.DataFrame({"feature": list(importance.keys()), "importance": list(importance.values())})
imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
display_names = {f: display_name(f) for f in imp_df["feature"]}

top_n = st.slider("Show top N features", min_value=5, max_value=len(imp_df), value=15)
st.plotly_chart(feature_importance_bar(imp_df, top_n=top_n, display_names=display_names), width="stretch", key="perf_feature_importance")

with st.expander("Predicted class distribution on the notebook's real test-set submission"):
    st.caption(
        "For reference, this is the class distribution the deployed pipeline produced on the "
        "competition's actual held-out test set (`assets/submission_lightgbm.csv`), vs. the "
        "training label distribution — a sanity check against label-shift bugs, not a metric "
        "computed by this app."
    )
    st.markdown(
        """
| Grade | Test predictions | Training labels |
|---|---|---|
| 1 | 7.96% | 9.64% |
| 2 | 63.66% | 56.89% |
| 3 | 28.38% | 33.47% |
"""
    )
