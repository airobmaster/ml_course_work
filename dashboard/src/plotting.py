"""Plotly chart builders. Functions take already-aggregated DataFrames
(produced by src.queries) and return a go.Figure — no raw-row plotting,
except the numeric box plot which needs per-row (feature, target) pairs."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DAMAGE_COLORS = {1: "#2ca02c", 2: "#f2c744", 3: "#d62728"}
DAMAGE_LABELS = {1: "Grade 1 (low)", 2: "Grade 2 (medium)", 3: "Grade 3 (severe)"}


def damage_distribution_pie(df: pd.DataFrame) -> go.Figure:
    fig = px.pie(
        df,
        names=df["damage_grade"].map(DAMAGE_LABELS),
        values="buildings",
        color=df["damage_grade"],
        color_discrete_map=DAMAGE_COLORS,
        hole=0.4,
    )
    fig.update_traces(textinfo="percent+label")
    return fig


def value_counts_bar(df: pd.DataFrame, value_col: str = "value") -> go.Figure:
    df = df.sort_values("buildings", ascending=True)
    fig = px.bar(
        df,
        x="buildings",
        y=df[value_col].astype(str),
        orientation="h",
        text=df["pct"].round(1).astype(str) + "%",
    )
    fig.update_layout(yaxis_title=None, xaxis_title="Buildings")
    return fig


def numeric_histogram(df: pd.DataFrame, value_col: str = "value") -> go.Figure:
    fig = px.bar(df.sort_values(value_col), x=value_col, y="buildings")
    fig.update_layout(xaxis_title=None, yaxis_title="Buildings")
    return fig


def stacked_damage_bar(crosstab_pct: pd.DataFrame) -> go.Figure:
    """crosstab_pct: index=feature value, columns=damage_grade (1/2/3), values=row %."""
    fig = go.Figure()
    for grade in sorted(crosstab_pct.columns):
        values = crosstab_pct[grade]
        fig.add_bar(
            name=DAMAGE_LABELS.get(grade, str(grade)),
            x=crosstab_pct.index.astype(str),
            y=values,
            marker_color=DAMAGE_COLORS.get(grade),
            text=[f"{v:.1f}%" if v >= 3 else "" for v in values],
            textposition="inside",
            insidetextanchor="middle",
            textfont_color="white",
        )
    fig.update_layout(
        barmode="stack",
        yaxis_title="% of buildings",
        xaxis_title=None,
        legend_title=None,
    )
    return fig


def damage_heatmap(crosstab_pct: pd.DataFrame) -> go.Figure:
    fig = px.imshow(
        crosstab_pct.values,
        x=[DAMAGE_LABELS.get(c, str(c)) for c in crosstab_pct.columns],
        y=crosstab_pct.index.astype(str),
        color_continuous_scale="Reds",
        text_auto=".1f",
        aspect="auto",
    )
    fig.update_layout(coloraxis_colorbar_title="%")
    return fig


def numeric_box_by_damage(df: pd.DataFrame, feature: str, target_col: str = "damage_grade") -> go.Figure:
    """Distribution of a numeric feature's raw values, split by damage grade -
    mirrors the boxplot-per-grade used in the source notebook's EDA."""
    fig = go.Figure()
    for grade in sorted(df[target_col].dropna().unique()):
        fig.add_trace(
            go.Box(
                y=df.loc[df[target_col] == grade, feature],
                name=DAMAGE_LABELS.get(grade, str(grade)),
                marker_color=DAMAGE_COLORS.get(grade),
                boxpoints=False,
            )
        )
    fig.update_layout(showlegend=False, yaxis_title=feature, xaxis_title=None)
    return fig


def probability_bar(probabilities: dict) -> go.Figure:
    """probabilities: {1: p1, 2: p2, 3: p3} -> horizontal bar of predicted
    class probabilities, used on the Predict Damage page."""
    grades = sorted(probabilities.keys())
    fig = go.Figure(
        go.Bar(
            x=[probabilities[g] * 100 for g in grades],
            y=[DAMAGE_LABELS.get(g, str(g)) for g in grades],
            orientation="h",
            marker_color=[DAMAGE_COLORS.get(g) for g in grades],
            text=[f"{probabilities[g] * 100:.1f}%" for g in grades],
            textposition="outside",
        )
    )
    fig.update_layout(
        xaxis_title="Probability (%)",
        yaxis_title=None,
        xaxis_range=[0, 100],
        showlegend=False,
    )
    return fig


def confusion_matrix_heatmap(cm, labels) -> go.Figure:
    """cm: 2D list/array of counts, labels: class labels in row/col order."""
    label_names = [DAMAGE_LABELS.get(l, str(l)) for l in labels]
    fig = px.imshow(
        cm,
        x=label_names,
        y=label_names,
        color_continuous_scale="Blues",
        text_auto=",d",
        aspect="auto",
        labels=dict(x="Predicted", y="Actual", color="Buildings"),
    )
    return fig


def feature_importance_bar(df: pd.DataFrame, top_n: int = 15, display_names: dict | None = None) -> go.Figure:
    """df: columns `feature`, `importance` (or `importance_pct`) -- global
    LightGBM feature importance, highest first."""
    top = df.head(top_n).sort_values("importance", ascending=True)
    names = top["feature"]
    if display_names:
        names = names.map(lambda c: display_names.get(c, c))
    fig = px.bar(top, x="importance", y=names, orientation="h")
    fig.update_layout(yaxis_title=None, xaxis_title="Importance (gain)")
    return fig


def shap_bar(df: pd.DataFrame, top_n: int = 15, display_names: dict | None = None) -> go.Figure:
    """df: columns `feature`, `value`, `shap_value` -- one prediction's SHAP
    contributions, colored by whether they push toward (red) or away from
    (blue) the predicted class."""
    top = df.head(top_n).sort_values("shap_value", key=lambda s: s.abs())
    labels = [
        f"{(display_names or {}).get(f, f)} = {v}" for f, v in zip(top["feature"], top["value"])
    ]
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in top["shap_value"]]
    fig = go.Figure(
        go.Bar(x=top["shap_value"], y=labels, orientation="h", marker_color=colors)
    )
    fig.update_layout(
        xaxis_title="SHAP value (impact on predicted class)",
        yaxis_title=None,
    )
    return fig


def whatif_probability_chart(df: pd.DataFrame, feature_display_name: str = "Value") -> go.Figure:
    """df from src.prediction.what_if_single_feature: columns `label`,
    `P(Grade 1)`, `P(Grade 2)`, `P(Grade 3)` -- stacked bar across every
    candidate value of the varied feature."""
    fig = go.Figure()
    for grade, col in zip((1, 2, 3), ("P(Grade 1)", "P(Grade 2)", "P(Grade 3)")):
        fig.add_bar(
            name=DAMAGE_LABELS[grade],
            x=df["label"].astype(str),
            y=df[col] * 100,
            marker_color=DAMAGE_COLORS[grade],
            text=[f"{v * 100:.0f}%" if v >= 0.03 else "" for v in df[col]],
            textposition="inside",
            insidetextanchor="middle",
            textfont_color="white",
        )
    fig.update_layout(
        barmode="stack",
        yaxis_title="Probability (%)",
        xaxis_title=feature_display_name,
        legend_title=None,
    )
    return fig


def scenario_comparison_chart(df: pd.DataFrame) -> go.Figure:
    """df from src.prediction.compare_scenarios: columns `scenario`,
    `P(Grade 1)`, `P(Grade 2)`, `P(Grade 3)` -- grouped bar, one group per
    scenario, comparing baseline vs each what-if scenario."""
    fig = go.Figure()
    for grade, col in zip((1, 2, 3), ("P(Grade 1)", "P(Grade 2)", "P(Grade 3)")):
        fig.add_bar(
            name=DAMAGE_LABELS[grade],
            x=df["scenario"],
            y=df[col] * 100,
            marker_color=DAMAGE_COLORS[grade],
            text=[f"{v * 100:.0f}%" for v in df[col]],
            textposition="outside",
        )
    fig.update_layout(
        barmode="group",
        yaxis_title="Probability (%)",
        xaxis_title=None,
        legend_title=None,
    )
    return fig


def geo_sunburst(nodes: pd.DataFrame) -> go.Figure:
    """nodes needs columns: id, label, parent, buildings, avg_damage_grade, hover
    (a pre-built breadcrumb string showing the full id hierarchy, as in the
    source notebook)."""
    fig = go.Figure(
        go.Sunburst(
            ids=nodes["id"],
            labels=nodes["label"],
            parents=nodes["parent"],
            values=nodes["buildings"],
            branchvalues="total",
            marker=dict(
                colors=nodes["avg_damage_grade"],
                colorscale="Reds",
                cmin=1,
                cmax=3,
                colorbar=dict(title="Avg<br>Damage<br>Grade"),
            ),
            customdata=nodes[["hover", "buildings", "avg_damage_grade"]],
            hovertemplate=(
                "%{customdata[0]}<br>"
                "Buildings: %{customdata[1]:,.0f}<br>"
                "Avg damage grade: %{customdata[2]:.2f}"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(margin=dict(t=10, l=10, r=10, b=10))
    return fig
