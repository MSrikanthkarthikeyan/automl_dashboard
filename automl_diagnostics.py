"""
automl_diagnostics.py
════════════════════════════════════════════════════════════════════════
Module 3 — Metric-Safe Diagnostic Visualizer
Replaces: plot_correlation_heatmap(), plot_feature_importance(),
          plot_roc_curves(), plot_results_bar()

Fixes addressed:
  ✔ Metric KeyError — all look-ups use .get() with safe fallbacks
  ✔ Classification  — Confusion Matrix + multi-model ROC Curves
  ✔ Regression      — Residual vs Fitted + Actual vs Predicted
  ✔ Shared          — Correlation heatmap, Feature importance,
                       Model comparison bar, 2-feature scatter dropdown
════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import label_binarize

# ─────────────────────────────────────────────────────────────────────
# SHARED THEME
# ─────────────────────────────────────────────────────────────────────

PLOTLY_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(10,10,15,0)",
    plot_bgcolor="rgba(10,10,15,0)",
    font=dict(family="JetBrains Mono, monospace", color="#9ca3af", size=11),
    margin=dict(l=50, r=30, t=55, b=45),
)

PALETTE = [
    "#7c3aed", "#00d4ff", "#f59e0b", "#10b981",
    "#ef4444", "#a78bfa", "#34d399", "#fbbf24",
    "#f472b6", "#60a5fa",
]

_HEATMAP_SCALE = [
    [0.00, "#1e0048"],
    [0.25, "#4c1d95"],
    [0.50, "#111118"],
    [0.75, "#0369a1"],
    [1.00, "#00d4ff"],
]


# ─────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────

def _dense(X) -> np.ndarray:
    """Convert sparse matrix to dense array for visualizations."""
    return X.toarray() if sp.issparse(X) else np.asarray(X)


def _get_proba(model, X_te) -> np.ndarray | None:
    """Return predict_proba array or None if unavailable."""
    try:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X_te)
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────
# 3-A. CORRELATION HEATMAP
# ─────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, target_col: str) -> None:
    """Interactive Pearson correlation heatmap for all numeric columns."""
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] < 2:
        st.info("Need at least 2 numeric columns to draw a correlation heatmap.")
        return

    corr = num_df.corr().round(3)
    z    = corr.values
    cols = corr.columns.tolist()

    fig = go.Figure(go.Heatmap(
        z=z,
        x=cols,
        y=cols,
        colorscale=_HEATMAP_SCALE,
        zmid=0,
        zmin=-1, zmax=1,
        text=np.round(z, 2),
        texttemplate="%{text}",
        textfont=dict(size=9),
        hovertemplate="<b>%{y} × %{x}</b><br>r = %{z:.3f}<extra></extra>",
        showscale=True,
        colorbar=dict(thickness=14, len=0.9),
    ))
    fig.update_layout(
        title="Pearson Correlation Matrix",
        height=max(420, len(cols) * 28 + 80),
        **PLOTLY_DARK,
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# 3-B. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────

def plot_feature_importance(
    model,
    feature_names: list[str],
    model_name: str,
    top_n: int = 25,
) -> None:
    """
    Horizontal bar chart.  Supports:
      • tree-based models  → feature_importances_
      • linear models      → coef_ (L2 norm over multi-class rows)
    """
    importances: np.ndarray | None = None

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_).ravel()
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef).ravel()

    if importances is None or len(importances) == 0:
        st.info(f"Feature importance is not available for **{model_name}**.")
        return

    # Guard: align lengths
    n = min(len(importances), len(feature_names))
    importances = importances[:n]
    names       = feature_names[:n]

    fi_df = (
        pd.DataFrame({"Feature": names, "Importance": importances})
        .sort_values("Importance", ascending=True)
        .tail(top_n)
    )

    fig = go.Figure(go.Bar(
        x=fi_df["Importance"],
        y=fi_df["Feature"],
        orientation="h",
        marker=dict(
            color=fi_df["Importance"],
            colorscale=[[0, "#4c1d95"], [0.5, "#7c3aed"], [1, "#00d4ff"]],
            showscale=True,
            colorbar=dict(thickness=12, len=0.8),
        ),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.5f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Feature Importance — {model_name} (top {top_n})",
        xaxis_title="Importance",
        yaxis=dict(tickfont=dict(size=10)),
        height=max(320, len(fi_df) * 22 + 100),
        **PLOTLY_DARK,
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# 3-C. MODEL COMPARISON BAR CHART
# ─────────────────────────────────────────────────────────────────────

def plot_results_bar(
    metrics_df: pd.DataFrame,
    display_cols: list[str],
    primary_metric: str,
) -> None:
    """Grouped bar chart — all metrics side-by-side per model."""
    metric_cols = [c for c in display_cols if c != "Model"]
    fig = go.Figure()

    for i, col in enumerate(metric_cols):
        if col not in metrics_df.columns:
            continue
        fig.add_trace(go.Bar(
            name=col,
            x=metrics_df["Model"],
            y=metrics_df[col],
            marker_color=PALETTE[i % len(PALETTE)],
            hovertemplate=f"<b>%{{x}}</b><br>{col}: %{{y:.4f}}<extra></extra>",
        ))

    fig.update_layout(
        barmode="group",
        title="Model Comparison — All Metrics",
        xaxis_title="Model",
        yaxis_title="Score",
        height=430,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **PLOTLY_DARK,
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# 3-D. CLASSIFICATION DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    results: list[dict],
    best_name: str,
    le_target=None,
) -> None:
    """Plotly heatmap confusion matrix for the best-performing model."""
    best = next((r for r in results if r["Model"] == best_name), None)
    if best is None:
        return

    y_te   = best.get("_y_te",   [])
    y_pred = best.get("_y_pred", [])

    if len(y_te) == 0 or len(y_pred) == 0:
        st.info("Confusion matrix data not available.")
        return

    labels = np.unique(np.concatenate([y_te, y_pred]))
    cm     = confusion_matrix(y_te, y_pred, labels=labels)

    # Decode labels if encoder was provided
    tick_labels = (
        le_target.inverse_transform(labels.astype(int)).tolist()
        if le_target is not None
        else labels.tolist()
    )
    tick_labels = [str(t) for t in tick_labels]

    # Normalised version for colour; raw counts for text
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig = go.Figure(go.Heatmap(
        z=cm_norm,
        x=tick_labels,
        y=tick_labels,
        text=cm,
        texttemplate="%{text}",
        textfont=dict(size=12, color="white"),
        colorscale=[[0, "#0a0a0f"], [0.5, "#4c1d95"], [1, "#7c3aed"]],
        hovertemplate=(
            "Actual: <b>%{y}</b><br>"
            "Predicted: <b>%{x}</b><br>"
            "Count: %{text}<br>"
            "Row %: %{z:.1%}<extra></extra>"
        ),
        showscale=True,
        colorbar=dict(thickness=14, tickformat=".0%"),
    ))
    fig.update_layout(
        title=f"Confusion Matrix — {best_name}",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        yaxis_autorange="reversed",
        height=max(380, len(labels) * 50 + 120),
        **PLOTLY_DARK,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_roc_curves(results: list[dict]) -> None:
    """
    Multi-model ROC curves on a single chart.
    Binary: single curve per model.
    Multiclass: one-vs-rest curves shown with dashed lines.
    """
    fig = go.Figure()

    # Random baseline
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(dash="dot", color="#374151", width=1.2),
        name="Random baseline",
        hoverinfo="skip",
    ))

    plotted = 0
    for i, res in enumerate(results):
        model  = res.get("_model")
        X_te   = res.get("_X_te")
        y_te   = res.get("_y_te", np.array([]))

        if model is None or X_te is None or len(y_te) == 0:
            continue

        y_prob = _get_proba(model, X_te)
        if y_prob is None:
            continue

        classes = np.unique(y_te)
        color   = PALETTE[i % len(PALETTE)]

        if len(classes) == 2:
            fpr, tpr, _ = roc_curve(y_te, y_prob[:, 1])
            roc_val     = auc(fpr, tpr)
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{res['Model']} (AUC={roc_val:.3f})",
                line=dict(color=color, width=2.2),
                hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
            ))
            plotted += 1
        else:
            y_bin = label_binarize(y_te, classes=classes)
            for j, cls_label in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_bin[:, j], y_prob[:, j])
                ra          = auc(fpr, tpr)
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines",
                    name=f"{res['Model']} cls={cls_label} (AUC={ra:.3f})",
                    line=dict(color=color, width=1.5, dash="dot"),
                    hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
                ))
            plotted += 1

    if plotted == 0:
        st.info("ROC curves require `predict_proba` support. No eligible models found.")
        return

    fig.update_layout(
        title="ROC Curves — All Models",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[-0.02, 1.02]),
        yaxis=dict(range=[-0.02, 1.05]),
        height=490,
        legend=dict(font=dict(size=9)),
        **PLOTLY_DARK,
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# 3-E. REGRESSION DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────

def plot_residuals_vs_fitted(results: list[dict]) -> None:
    """
    Residual vs Fitted subplot grid — one panel per model (max 6).
    """
    top = results[:6]
    n   = len(top)
    if n == 0:
        return

    cols_n = min(n, 2)
    rows_n = (n + 1) // cols_n

    titles = [r["Model"] for r in top]
    fig    = make_subplots(
        rows=rows_n, cols=cols_n,
        subplot_titles=titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    for idx, res in enumerate(top):
        model  = res.get("_model")
        X_te   = res.get("_X_te")
        y_te   = res.get("_y_te", np.array([]))

        row = idx // cols_n + 1
        col = idx % cols_n + 1

        if model is None or X_te is None or len(y_te) == 0:
            continue

        y_pred    = model.predict(X_te)
        residuals = y_te - y_pred
        color     = PALETTE[idx % len(PALETTE)]

        fig.add_trace(
            go.Scatter(
                x=y_pred, y=residuals,
                mode="markers",
                marker=dict(color=color, opacity=0.55, size=5),
                name=res["Model"],
                hovertemplate="Fitted: %{x:.3f}<br>Residual: %{y:.3f}<extra></extra>",
            ),
            row=row, col=col,
        )
        # Zero-line
        fig.add_hline(
            y=0, line_dash="dash", line_color="#374151", line_width=1,
            row=row, col=col,
        )

    fig.update_layout(
        title="Residuals vs Fitted Values",
        height=380 * rows_n,
        showlegend=False,
        **PLOTLY_DARK,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_actual_vs_predicted(results: list[dict], best_name: str) -> None:
    """
    Scatter of Actual vs Predicted for the best model, with a perfect-
    prediction diagonal line.
    """
    best = next((r for r in results if r["Model"] == best_name), None)
    if best is None:
        return

    model = best.get("_model")
    X_te  = best.get("_X_te")
    y_te  = best.get("_y_te", np.array([]))

    if model is None or X_te is None or len(y_te) == 0:
        return

    y_pred = model.predict(X_te)
    lo     = float(min(y_te.min(), y_pred.min()))
    hi     = float(max(y_te.max(), y_pred.max()))

    fig = go.Figure()

    # Perfect prediction line
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi],
        mode="lines",
        line=dict(dash="dash", color="#374151", width=1.5),
        name="Perfect prediction",
    ))

    # Scatter
    fig.add_trace(go.Scatter(
        x=y_te.tolist(), y=y_pred.tolist(),
        mode="markers",
        marker=dict(color="#7c3aed", opacity=0.6, size=5),
        name=best_name,
        hovertemplate="Actual: %{x:.3f}<br>Predicted: %{y:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title=f"Actual vs Predicted — {best_name}",
        xaxis_title="Actual",
        yaxis_title="Predicted",
        height=440,
        **PLOTLY_DARK,
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# 3-F. 2-FEATURE INTERACTIVE SCATTER
# ─────────────────────────────────────────────────────────────────────

def plot_feature_scatter(
    df: pd.DataFrame,
    target_col: str,
    task: str,
    le_target=None,
) -> None:
    """
    Two st.selectbox dropdowns let the user pick any two numeric features.
    Points are coloured by the target variable (class for classification,
    continuous gradient for regression).
    """
    num_feats = df.select_dtypes(include=np.number).columns.tolist()
    # Remove target from feature options
    num_feats = [c for c in num_feats if c != target_col]

    if len(num_feats) < 2:
        st.info("Need at least 2 numeric feature columns for the scatter chart.")
        return

    c1, c2 = st.columns(2)
    with c1:
        feat_x = st.selectbox("X-axis feature", num_feats, index=0, key="scatter_x")
    with c2:
        default_y = num_feats[1] if len(num_feats) > 1 else num_feats[0]
        feat_y = st.selectbox("Y-axis feature", num_feats, index=1, key="scatter_y")

    plot_df = df[[feat_x, feat_y, target_col]].dropna()

    if task == "classification":
        color_col = target_col
        # Decode if encoder exists
        if le_target is not None and pd.api.types.is_numeric_dtype(plot_df[target_col]):
            try:
                plot_df = plot_df.copy()
                plot_df[target_col] = le_target.inverse_transform(
                    plot_df[target_col].astype(int)
                )
            except Exception:
                pass
        plot_df[color_col] = plot_df[color_col].astype(str)

        fig = px.scatter(
            plot_df, x=feat_x, y=feat_y, color=color_col,
            color_discrete_sequence=PALETTE,
            opacity=0.7,
            hover_data={feat_x: ":.3f", feat_y: ":.3f", color_col: True},
        )
    else:
        fig = px.scatter(
            plot_df, x=feat_x, y=feat_y, color=target_col,
            color_continuous_scale=["#4c1d95", "#7c3aed", "#00d4ff", "#f59e0b"],
            opacity=0.7,
            hover_data={feat_x: ":.3f", feat_y: ":.3f", target_col: ":.3f"},
        )

    fig.update_traces(marker=dict(size=5))
    fig.update_layout(
        title=f"Feature Scatter: {feat_x}  ×  {feat_y} | coloured by {target_col}",
        height=450,
        **PLOTLY_DARK,
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# 3-G. FULL DIAGNOSTIC PANEL (orchestrator called from main app)
# ─────────────────────────────────────────────────────────────────────

def render_diagnostic_panel(
    results:      list[dict],
    df:           pd.DataFrame,
    target_col:   str,
    task:         str,
    feature_names: list[str],
    best_name:    str,
    le_target=None,
) -> None:
    """
    Renders all diagnostic tabs in the Streamlit UI.
    Call this from the main app after training is complete.
    """
    if task == "classification":
        tabs = st.tabs([
            "🔥 Correlation",
            "📊 Feature Importance",
            "🧩 Confusion Matrix",
            "📈 ROC Curves",
            "🔵 Scatter Explorer",
        ])
        with tabs[0]:
            plot_correlation_heatmap(df, target_col)
        with tabs[1]:
            best_res = next((r for r in results if r["Model"] == best_name), results[0])
            plot_feature_importance(best_res["_model"], feature_names, best_name)
        with tabs[2]:
            plot_confusion_matrix(results, best_name, le_target)
        with tabs[3]:
            plot_roc_curves(results)
        with tabs[4]:
            plot_feature_scatter(df, target_col, task, le_target)

    else:
        tabs = st.tabs([
            "🔥 Correlation",
            "📊 Feature Importance",
            "📉 Residuals",
            "🎯 Actual vs Predicted",
            "🔵 Scatter Explorer",
        ])
        with tabs[0]:
            plot_correlation_heatmap(df, target_col)
        with tabs[1]:
            best_res = next((r for r in results if r["Model"] == best_name), results[0])
            plot_feature_importance(best_res["_model"], feature_names, best_name)
        with tabs[2]:
            plot_residuals_vs_fitted(results)
        with tabs[3]:
            plot_actual_vs_predicted(results, best_name)
        with tabs[4]:
            plot_feature_scatter(df, target_col, task, le_target)
