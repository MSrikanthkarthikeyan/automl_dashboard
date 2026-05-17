"""
automl_dashboard.py  (v2 — refactored)
════════════════════════════════════════════════════════════════════════
AutoML Dashboard — Production-ready end-to-end ML pipeline
Stack : Streamlit · Pandas · Scikit-learn · XGBoost · LightGBM
        · CatBoost · SciPy (sparse) · Plotly

Modules wired in:
  automl_ingestion.py   — memory-safe ingestion & sanitization
  automl_models.py      — expanded model registry + isolated training
  automl_diagnostics.py — full diagnostic visualizer

Run:
    pip install -r requirements.txt
    streamlit run automl_dashboard.py
════════════════════════════════════════════════════════════════════════
"""

# ── stdlib ────────────────────────────────────────────────────────────
import warnings

# ── third-party ───────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import scipy.sparse as sp
import streamlit as st
import plotly.express as px

# ── local modules ─────────────────────────────────────────────────────
from automl_ingestion import (
    ingest_data,
    auto_clean,
    HIGH_CARDINALITY_THRESHOLD,
)
from automl_models import (
    get_models,
    run_training,
    build_metrics_df,
    ALGO_NAMES_BY_TASK,
    XGBOOST_OK,
    LGBM_OK,
    CATBOOST_OK,
)
from automl_diagnostics import (
    render_diagnostic_panel,
    plot_results_bar,
    PLOTLY_DARK,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoML Dashboard v2",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"]  { font-family: 'Syne', sans-serif; }
.stApp { background: #0a0a0f; color: #e8e8f0; }

.main-header {
    font-family: 'Syne', sans-serif; font-weight: 800;
    font-size: 2.8rem; letter-spacing: -1px;
    background: linear-gradient(135deg,#00d4ff 0%,#7c3aed 50%,#f59e0b 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: .2rem;
}
.sub-header {
    color: #6b7280; font-size: .9rem; letter-spacing: .1em;
    text-transform: uppercase; margin-bottom: 1.8rem;
    font-family: 'JetBrains Mono', monospace;
}
.metric-card {
    background: linear-gradient(135deg,#111118,#1a1a2e);
    border: 1px solid #2d2d4a; border-radius: 12px;
    padding: 1.1rem 1.4rem; margin: .35rem 0;
}
.metric-label {
    font-family: 'JetBrains Mono', monospace; font-size: .7rem;
    color: #6b7280; text-transform: uppercase; letter-spacing: .12em;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace; font-size: 1.7rem;
    font-weight: 700; color: #00d4ff; margin-top: .2rem;
}
.section-tag {
    display: inline-block; background: #7c3aed22;
    border: 1px solid #7c3aed55; color: #a78bfa;
    font-family: 'JetBrains Mono', monospace; font-size: .7rem;
    padding: .2rem .7rem; border-radius: 4px; letter-spacing: .1em;
    text-transform: uppercase; margin-bottom: .5rem;
}
div[data-testid="stSidebar"] {
    background: #07070d; border-right: 1px solid #1e1e30;
}
div[data-testid="stSidebar"] label {
    color: #9ca3af !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .8rem !important;
}
.stButton > button {
    background: linear-gradient(135deg,#7c3aed,#4f46e5);
    color: white; border: none; border-radius: 8px;
    font-family: 'Syne', sans-serif; font-weight: 700;
    padding: .6rem 1.8rem; width: 100%; transition: all .2s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg,#6d28d9,#4338ca);
    transform: translateY(-1px); box-shadow: 0 8px 24px #7c3aed44;
}
.best-model-highlight {
    background: linear-gradient(135deg,#064e3b22,#065f4622);
    border: 1px solid #10b98155; border-radius: 8px;
    padding: 1rem 1.4rem; margin: 1rem 0;
}
.best-model-title {
    font-family: 'JetBrains Mono', monospace; font-size: .75rem;
    color: #10b981; text-transform: uppercase; letter-spacing: .1em;
}
.best-model-name {
    font-family: 'Syne', sans-serif; font-size: 1.5rem;
    font-weight: 800; color: #34d399; margin-top: .2rem;
}
.task-pill {
    display: inline-block; padding: .25rem .9rem; border-radius: 999px;
    font-family: 'JetBrains Mono', monospace;
    font-size: .78rem; font-weight: 700; letter-spacing: .08em;
}
.task-classification {
    background: #1e40af22; border: 1px solid #3b82f655; color: #60a5fa;
}
.task-regression {
    background: #78350f22; border: 1px solid #f59e0b55; color: #fbbf24;
}
hr.divider { border: none; border-top: 1px solid #1e1e30; margin: 1.8rem 0; }
.info-box {
    background: #0c0c18; border-left: 3px solid #7c3aed;
    padding: .8rem 1rem; border-radius: 0 8px 8px 0;
    font-family: 'JetBrains Mono', monospace; font-size: .82rem;
    color: #9ca3af; margin: .8rem 0;
}
.avail-ok { color: #34d399; font-size: .75rem; }
.avail-no { color: #4b5563; font-size: .75rem; text-decoration: line-through; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────

def card(label: str, value: str) -> None:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def section_tag(text: str) -> None:
    st.markdown(f'<div class="section-tag">⬡ {text}</div>', unsafe_allow_html=True)


def hr() -> None:
    st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────
_defaults: dict = {
    "df":            None,
    "X":             None,
    "y":             None,
    "le_target":     None,
    "feature_names": [],
    "task":          None,
    "results":       None,
    "trained":       False,
    "dropped_cols":  [],
    "target_col":    None,
}
for k, v in _defaults.items():
    st.session_state.setdefault(k, v)


# ─────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="font-family:Syne,sans-serif;font-weight:800;'
        'font-size:1.3rem;color:#a78bfa;margin-bottom:1.5rem;">⚡ AutoML Controls</div>',
        unsafe_allow_html=True,
    )

    # ── Ingestion ────────────────────────────────
    st.markdown("##### 01 · Data Ingestion")
    uploaded_files = st.file_uploader(
        "Upload CSV / Parquet",
        type=["csv", "parquet"],
        accept_multiple_files=True,
        help="Multiple files are merged automatically on shared columns.",
    )
    url_input = st.text_input(
        "…or paste a dataset URL",
        placeholder="https://…/data.csv",
    )

    st.markdown("---")

    # ── Feature Engineering ──────────────────────
    st.markdown("##### 02 · Feature Engineering")
    encoding_method = st.radio(
        "Categorical Encoding",
        ["Label Encoding", "One-Hot Encoding"],
    )
    test_split = st.slider(
        "Test Split Ratio",
        min_value=0.10, max_value=0.40,
        value=0.20, step=0.05, format="%.2f",
    )

    st.markdown("---")

    # ── Algorithm Selection ──────────────────────
    st.markdown("##### 03 · Algorithm Selection")

    lib_badges = "  ".join([
        f"<span class='avail-ok'>✓ XGBoost</span>" if XGBOOST_OK
            else "<span class='avail-no'>✗ XGBoost</span>",
        f"<span class='avail-ok'>✓ LightGBM</span>" if LGBM_OK
            else "<span class='avail-no'>✗ LightGBM</span>",
        f"<span class='avail-ok'>✓ CatBoost</span>" if CATBOOST_OK
            else "<span class='avail-no'>✗ CatBoost</span>",
    ])
    st.markdown(lib_badges, unsafe_allow_html=True)

    auto_select = st.checkbox("Auto-Select All", value=True)

    all_sidebar_algos = sorted(set(
        ALGO_NAMES_BY_TASK["classification"] + ALGO_NAMES_BY_TASK["regression"]
    ))

    if auto_select:
        selected_algos = all_sidebar_algos
        st.caption(f"All {len(selected_algos)} algorithms queued ✓")
    else:
        selected_algos = st.multiselect(
            "Choose algorithms",
            options=all_sidebar_algos,
            default=["Random Forest", "Gradient Boosting"],
            help="Task-irrelevant algorithms are filtered out automatically at runtime.",
        )

    st.markdown("---")
    run_btn = st.button("🚀  Run AutoML Pipeline", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">AutoML Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Ingest · Sanitize · Train · Evaluate · Diagnose  —  v2 refactor'
    '</div>',
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════
# STEP 1 — DATA INGESTION
# ══════════════════════════════════════════════
section_tag("STEP 01 — DATA INGESTION")

if uploaded_files or url_input.strip():
    df_loaded = ingest_data(uploaded_files, url_input)
    if df_loaded is not None:
        st.session_state.df = df_loaded

        c1, c2, c3, c4 = st.columns(4)
        with c1: card("Rows",         f"{df_loaded.shape[0]:,}")
        with c2: card("Columns",      f"{df_loaded.shape[1]}")
        with c3: card("Missing Vals", f"{df_loaded.isnull().sum().sum():,}")
        with c4: card("Memory",       f"{df_loaded.memory_usage(deep=True).sum() / 1e6:.1f} MB")

        with st.expander("📋  Preview Dataset (first 50 rows)", expanded=False):
            st.dataframe(df_loaded.head(50), use_container_width=True)
else:
    st.markdown(
        '<div class="info-box">'
        'Upload a CSV / Parquet file or enter a dataset URL in the sidebar to begin.'
        '</div>',
        unsafe_allow_html=True,
    )

df: pd.DataFrame | None = st.session_state.df

if df is not None:
    hr()

    # ══════════════════════════════════════════
    # STEP 2 — TARGET SELECTION
    # ══════════════════════════════════════════
    section_tag("STEP 02 — TARGET & AUTO-CLEANING")

    target_col = st.selectbox(
        "Select Target Variable",
        options=df.columns.tolist(),
        index=len(df.columns) - 1,
    )
    st.session_state.target_col = target_col

    n_null = int(df[target_col].isnull().sum())
    if n_null:
        st.info(
            f"ℹ️  **{target_col}** contains **{n_null:,}** null row(s). "
            f"These will be dropped before training."
        )

    # ══════════════════════════════════════════
    # RUN
    # ══════════════════════════════════════════
    if run_btn:
        if not selected_algos:
            st.error("Select at least one algorithm in the sidebar.")
            st.stop()

        with st.spinner("🧹  Sanitizing dataset…"):
            try:
                X, y, le_target, feature_names, task, dropped_cols = auto_clean(
                    df, target_col, encoding_method
                )
            except ValueError as exc:
                st.error(f"❌  Cleaning failed: {exc}")
                st.stop()

        # Persist in session
        st.session_state.X             = X
        st.session_state.y             = y
        st.session_state.le_target     = le_target
        st.session_state.feature_names = feature_names
        st.session_state.task          = task
        st.session_state.dropped_cols  = dropped_cols

        shape_str = (
            f"{X.shape[0]:,} rows × {X.shape[1]:,} features"
            + (" (sparse)" if sp.issparse(X) else " (dense)")
        )
        task_label = "Classification" if task == "classification" else "Regression"
        task_cls   = "task-classification" if task == "classification" else "task-regression"

        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.markdown(
                f'Detected task: <span class="task-pill {task_cls}">{task_label}</span>',
                unsafe_allow_html=True,
            )
        with col_b:
            st.caption(f"Matrix: {shape_str}")

        hr()
        section_tag("STEP 03+04 — TRAINING & EVALUATION")

        models = get_models(task, selected_algos)
        if not models:
            st.error(
                "No algorithms matched the detected task and your selection. "
                "Make sure classification or regression algorithms are selected."
            )
            st.stop()

        results = run_training(models, X, y, test_split, task)
        if not results:
            st.error("Every model failed. Review the error messages above.")
            st.stop()

        st.session_state.results = results
        st.session_state.trained = True

    # ══════════════════════════════════════════
    # RESULTS (re-rendered on every rerun)
    # ══════════════════════════════════════════
    if st.session_state.trained and st.session_state.results:
        results       = st.session_state.results
        task          = st.session_state.task
        feature_names = st.session_state.feature_names
        le_target     = st.session_state.le_target
        target_col    = st.session_state.target_col or target_col

        hr()
        section_tag("RESULTS — MODEL COMPARISON")

        # ── Metric table (R² and R2 both available internally)
        metrics_df, display_cols, primary = build_metrics_df(results, task)

        ascending = primary in ("RMSE", "MAE")
        best_row  = metrics_df.iloc[0] if ascending else metrics_df.iloc[-1]
        best_name = best_row["Model"]

        # Use .get() to guard against any residual KeyError
        best_score = best_row.get(primary, best_row.get("R2", float("nan")))

        st.markdown(
            f'<div class="best-model-highlight">'
            f'<div class="best-model-title">🏆 Best Performing Model</div>'
            f'<div class="best-model-name">{best_name}</div>'
            f'<div style="font-family:JetBrains Mono,monospace;font-size:.8rem;'
            f'color:#6ee7b7;margin-top:.3rem;">'
            f'{primary} = {best_score:.4f}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        fmt_dict = {c: "{:.4f}" for c in display_cols if c != "Model"}
        st.dataframe(
            metrics_df[display_cols]
            .style
            .format(fmt_dict)
            .background_gradient(
                subset=[primary],
                cmap="RdYlGn" if primary not in ("RMSE", "MAE") else "RdYlGn_r",
            ),
            use_container_width=True,
        )

        with st.expander("⏱  Training times", expanded=False):
            st.dataframe(
                pd.DataFrame([
                    {"Model": r["Model"], "Train Time (s)": r.get("_elapsed_s", "—")}
                    for r in results
                ]),
                use_container_width=True,
            )

        plot_results_bar(metrics_df, display_cols, primary)

        hr()

        # ══════════════════════════════════════
        # STEP 5 — DIAGNOSTIC PANELS
        # ══════════════════════════════════════
        section_tag("STEP 05 — DIAGNOSTICS & VISUALIZATIONS")

        render_diagnostic_panel(
            results        = results,
            df             = df,
            target_col     = target_col,
            task           = task,
            feature_names  = feature_names,
            best_name      = best_name,
            le_target      = le_target,
        )

        hr()

        # ══════════════════════════════════════
        # TARGET DISTRIBUTION
        # ══════════════════════════════════════
        section_tag("TARGET DISTRIBUTION")
        y_series = st.session_state.y

        if task == "classification":
            vc = y_series.value_counts().reset_index()
            vc.columns = ["Class", "Count"]
            if le_target is not None:
                try:
                    vc["Class"] = le_target.inverse_transform(vc["Class"].astype(int))
                except Exception:
                    pass
            vc["Class"] = vc["Class"].astype(str)
            fig = px.bar(
                vc, x="Class", y="Count", color="Count",
                color_continuous_scale=["#4c1d95", "#7c3aed", "#00d4ff"],
            )
        else:
            fig = px.histogram(
                y_series, nbins=40,
                color_discrete_sequence=["#7c3aed"],
            )
            fig.update_layout(xaxis_title=target_col, yaxis_title="Count")

        fig.update_layout(
            title="Target Variable Distribution",
            height=360,
            **PLOTLY_DARK,
        )
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:3rem;padding:1.5rem;
            border-top:1px solid #1e1e30;
            font-family:'JetBrains Mono',monospace;
            font-size:.72rem;color:#374151;">
    ⚡ AutoML Dashboard v2 —
    scikit-learn · XGBoost · LightGBM · CatBoost · SciPy · Streamlit
</div>
""", unsafe_allow_html=True)
