"""
AutoML Dashboard — Production-ready end-to-end ML pipeline
Author: Senior ML Engineer
Stack: Streamlit · Pandas · Scikit-learn · XGBoost · LightGBM · Plotly
"""

import io
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, label_binarize
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    r2_score, mean_squared_error, mean_absolute_error,
    roc_curve, auc,
)

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AutoML Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Dark industrial theme */
.stApp {
    background: #0a0a0f;
    color: #e8e8f0;
}

.main-header {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.8rem;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 50%, #f59e0b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}

.sub-header {
    color: #6b7280;
    font-size: 0.95rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 2rem;
    font-family: 'JetBrains Mono', monospace;
}

.metric-card {
    background: linear-gradient(135deg, #111118 0%, #1a1a2e 100%);
    border: 1px solid #2d2d4a;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.4rem 0;
}

.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}

.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #00d4ff;
    margin-top: 0.2rem;
}

.section-tag {
    display: inline-block;
    background: #7c3aed22;
    border: 1px solid #7c3aed55;
    color: #a78bfa;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    padding: 0.2rem 0.7rem;
    border-radius: 4px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.step-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: #7c3aed;
    color: white;
    border-radius: 50%;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    font-weight: 700;
    margin-right: 0.6rem;
}

div[data-testid="stSidebar"] {
    background: #07070d;
    border-right: 1px solid #1e1e30;
}

div[data-testid="stSidebar"] .stSelectbox label,
div[data-testid="stSidebar"] .stMultiSelect label,
div[data-testid="stSidebar"] .stSlider label,
div[data-testid="stSidebar"] .stCheckbox label {
    color: #9ca3af !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
}

.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    letter-spacing: 0.05em;
    padding: 0.6rem 1.8rem;
    transition: all 0.2s ease;
    width: 100%;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #6d28d9, #4338ca);
    transform: translateY(-1px);
    box-shadow: 0 8px 24px #7c3aed44;
}

.stDataFrame {
    border: 1px solid #2d2d4a;
    border-radius: 8px;
}

.best-model-highlight {
    background: linear-gradient(135deg, #064e3b22, #065f4622);
    border: 1px solid #10b98155;
    border-radius: 8px;
    padding: 1rem 1.4rem;
    margin: 1rem 0;
}

.best-model-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #10b981;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.best-model-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    color: #34d399;
    margin-top: 0.2rem;
}

.task-pill {
    display: inline-block;
    padding: 0.25rem 0.9rem;
    border-radius: 999px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
}

.task-classification {
    background: #1e40af22;
    border: 1px solid #3b82f655;
    color: #60a5fa;
}

.task-regression {
    background: #78350f22;
    border: 1px solid #f59e0b55;
    color: #fbbf24;
}

hr.divider {
    border: none;
    border-top: 1px solid #1e1e30;
    margin: 2rem 0;
}

.info-box {
    background: #0c0c18;
    border-left: 3px solid #7c3aed;
    padding: 0.8rem 1rem;
    border-radius: 0 8px 8px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #9ca3af;
    margin: 0.8rem 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPER UTILITIES
# ─────────────────────────────────────────────

PLOTLY_DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(10,10,15,0)",
    plot_bgcolor="rgba(10,10,15,0)",
    font=dict(family="JetBrains Mono, monospace", color="#9ca3af"),
    margin=dict(l=40, r=20, t=50, b=40),
)

COLOR_SEQUENCE = [
    "#7c3aed", "#00d4ff", "#f59e0b", "#10b981",
    "#ef4444", "#a78bfa", "#34d399", "#fbbf24",
]


def card(label: str, value: str):
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def section_tag(text: str):
    st.markdown(f'<div class="section-tag">⬡ {text}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 1. DATA INGESTION
# ─────────────────────────────────────────────

def load_dataframe(source) -> pd.DataFrame:
    """Load a single CSV or Parquet source (file-like or URL string)."""
    if isinstance(source, str):
        url = source.strip()
        if url.endswith(".parquet"):
            return pd.read_parquet(url)
        return pd.read_csv(url)
    name = source.name.lower()
    if name.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(source.read()))
    return pd.read_csv(source)


def ingest_data(uploaded_files, url_input: str) -> pd.DataFrame | None:
    frames = []

    if uploaded_files:
        for f in uploaded_files:
            try:
                frames.append(load_dataframe(f))
            except Exception as e:
                st.error(f"❌  Could not read **{f.name}**: {e}")

    if url_input.strip():
        try:
            frames.append(load_dataframe(url_input.strip()))
        except Exception as e:
            st.error(f"❌  Could not fetch URL: {e}")

    if not frames:
        return None
    if len(frames) == 1:
        return frames[0]

    # Merge multiple frames — align on common columns, concat otherwise
    common_cols = set(frames[0].columns)
    for f in frames[1:]:
        common_cols &= set(f.columns)

    if common_cols:
        return pd.concat([f[list(common_cols)] for f in frames], ignore_index=True)
    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────
# 2. AUTO-CLEANING
# ─────────────────────────────────────────────

def detect_task(y: pd.Series) -> str:
    """Regression if target is numeric with many unique values, else classification."""
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 15:
        return "regression"
    return "classification"


def auto_clean(df: pd.DataFrame, target_col: str, encoding: str) -> tuple:
    """
    Returns (X_processed, y_encoded, label_encoder | None, feature_names, task_type)
    """
    df = df.copy()

    # Separate target
    y_raw = df[target_col].copy()
    X_raw = df.drop(columns=[target_col])

    task = detect_task(y_raw)

    # Encode target if classification
    le_target = None
    if task == "classification" and not pd.api.types.is_numeric_dtype(y_raw):
        le_target = LabelEncoder()
        y = pd.Series(le_target.fit_transform(y_raw.astype(str)), name=target_col)
    else:
        y = y_raw.reset_index(drop=True)

    # Split numeric / categorical features
    num_cols = X_raw.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X_raw.select_dtypes(exclude=np.number).columns.tolist()

    # Impute numerics
    if num_cols:
        num_imputer = SimpleImputer(strategy="median")
        X_num = pd.DataFrame(
            num_imputer.fit_transform(X_raw[num_cols]),
            columns=num_cols,
        )
    else:
        X_num = pd.DataFrame()

    # Impute + encode categoricals
    X_cat_parts = []
    cat_feature_names = []

    if cat_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_cat_imp = pd.DataFrame(
            cat_imputer.fit_transform(X_raw[cat_cols].astype(str)),
            columns=cat_cols,
        )

        if encoding == "Label Encoding":
            for col in cat_cols:
                le = LabelEncoder()
                X_cat_parts.append(
                    pd.Series(le.fit_transform(X_cat_imp[col]), name=col)
                )
                cat_feature_names.append(col)
        else:  # One-Hot
            dummies = pd.get_dummies(X_cat_imp, prefix=cat_cols)
            X_cat_parts.append(dummies)
            cat_feature_names.extend(dummies.columns.tolist())

    if X_cat_parts:
        X_cat_df = pd.concat(X_cat_parts, axis=1)
    else:
        X_cat_df = pd.DataFrame()

    X = pd.concat([X_num.reset_index(drop=True), X_cat_df.reset_index(drop=True)], axis=1)
    feature_names = X.columns.tolist()

    # Scale all numeric features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)

    return X_scaled, y.reset_index(drop=True), le_target, feature_names, task


# ─────────────────────────────────────────────
# 3. MODEL REGISTRY
# ─────────────────────────────────────────────

def get_models(task: str, selected_names: list) -> dict:
    mapping_clf = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "SVM": SVC(probability=True, random_state=42),
    }
    mapping_reg = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "SVM": SVR(),
    }

    if XGBOOST_AVAILABLE:
        mapping_clf["XGBoost"] = XGBClassifier(
            n_estimators=200, random_state=42, eval_metric="logloss",
            use_label_encoder=False, verbosity=0,
        )
        mapping_reg["XGBoost"] = XGBRegressor(
            n_estimators=200, random_state=42, verbosity=0,
        )

    if LGBM_AVAILABLE:
        mapping_clf["LightGBM"] = LGBMClassifier(
            n_estimators=200, random_state=42, verbose=-1,
        )
        mapping_reg["LightGBM"] = LGBMRegressor(
            n_estimators=200, random_state=42, verbose=-1,
        )

    mapping = mapping_clf if task == "classification" else mapping_reg
    return {k: v for k, v in mapping.items() if k in selected_names}


# ─────────────────────────────────────────────
# 4. TRAINING & EVALUATION
# ─────────────────────────────────────────────

def train_evaluate_classification(
    model, X_tr, X_te, y_tr, y_te, name: str
) -> dict:
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred, average="weighted", zero_division=0)

    # ROC-AUC: binary or multi-class OVR
    try:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_te)
            n_cls = len(np.unique(y_te))
            if n_cls == 2:
                roc = roc_auc_score(y_te, y_prob[:, 1])
            else:
                roc = roc_auc_score(y_te, y_prob, multi_class="ovr", average="weighted")
        else:
            roc = float("nan")
    except Exception:
        roc = float("nan")

    return {"Model": name, "Accuracy": acc, "F1 Score": f1, "ROC-AUC": roc,
            "_model": model}


def train_evaluate_regression(
    model, X_tr, X_te, y_tr, y_te, name: str
) -> dict:
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    r2 = r2_score(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    mae = mean_absolute_error(y_te, y_pred)

    return {"Model": name, "R²": r2, "RMSE": rmse, "MAE": mae,
            "_model": model}


def run_training(models: dict, X, y, test_size: float, task: str) -> list:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=42,
        stratify=y if task == "classification" else None,
    )

    results = []
    prog = st.progress(0, text="Training models…")
    total = len(models)

    for i, (name, model) in enumerate(models.items()):
        prog.progress((i) / total, text=f"Training  **{name}**…")
        if task == "classification":
            res = train_evaluate_classification(model, X_tr, X_te, y_tr, y_te, name)
        else:
            res = train_evaluate_regression(model, X_tr, X_te, y_tr, y_te, name)
        res["_X_tr"] = X_tr
        res["_X_te"] = X_te
        res["_y_tr"] = y_tr
        res["_y_te"] = y_te
        results.append(res)

    prog.progress(1.0, text="✅  All models trained!")
    return results


# ─────────────────────────────────────────────
# 5. VISUALIZATIONS
# ─────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, target: str):
    num_df = df.select_dtypes(include=np.number)
    if num_df.empty:
        st.info("No numeric columns available for correlation heatmap.")
        return

    corr = num_df.corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=[
            [0.0, "#1e0048"], [0.25, "#4c1d95"], [0.5, "#111118"],
            [0.75, "#0369a1"], [1.0, "#00d4ff"],
        ],
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        hovertemplate="<b>%{y} × %{x}</b><br>r = %{z:.3f}<extra></extra>",
        showscale=True,
    ))
    fig.update_layout(
        title="Correlation Matrix",
        height=500,
        **PLOTLY_DARK_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_feature_importance(model, feature_names: list, model_name: str):
    importances = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)

    if importances is None:
        st.info(f"Feature importance not available for {model_name}.")
        return

    fi_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=True)
        .tail(20)
    )

    fig = go.Figure(go.Bar(
        x=fi_df["Importance"],
        y=fi_df["Feature"],
        orientation="h",
        marker=dict(
            color=fi_df["Importance"],
            colorscale=[[0, "#4c1d95"], [0.5, "#7c3aed"], [1, "#00d4ff"]],
            showscale=False,
        ),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Feature Importance — {model_name}",
        xaxis_title="Importance",
        height=max(300, len(fi_df) * 24 + 80),
        **PLOTLY_DARK_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_roc_curves(results: list, task: str):
    """Plot ROC curves (classification) or Residuals (regression) for all models."""
    if task == "classification":
        fig = go.Figure()
        # diagonal
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines", line=dict(dash="dash", color="#374151", width=1),
            showlegend=False,
        ))

        for i, res in enumerate(results):
            model = res["_model"]
            X_te = res["_X_te"]
            y_te = res["_y_te"]

            if not hasattr(model, "predict_proba"):
                continue
            try:
                y_prob = model.predict_proba(X_te)
                n_cls = len(np.unique(y_te))
                if n_cls == 2:
                    fpr, tpr, _ = roc_curve(y_te, y_prob[:, 1])
                    roc_val = auc(fpr, tpr)
                    fig.add_trace(go.Scatter(
                        x=fpr, y=tpr, mode="lines",
                        name=f"{res['Model']} (AUC={roc_val:.3f})",
                        line=dict(color=COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)], width=2),
                    ))
                else:
                    y_bin = label_binarize(y_te, classes=np.unique(y_te))
                    for j in range(n_cls):
                        fpr, tpr, _ = roc_curve(y_bin[:, j], y_prob[:, j])
                        ra = auc(fpr, tpr)
                        fig.add_trace(go.Scatter(
                            x=fpr, y=tpr, mode="lines",
                            name=f"{res['Model']} cls-{j} (AUC={ra:.3f})",
                            line=dict(color=COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)], width=1.5, dash="dot"),
                        ))
            except Exception:
                continue

        fig.update_layout(
            title="ROC Curves — All Models",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=480,
            **PLOTLY_DARK_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        # Residual plots — one subplot per model (up to 4)
        top_results = results[:4]
        n = len(top_results)
        cols = 2
        rows = (n + 1) // cols

        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[r["Model"] for r in top_results],
        )
        for idx, res in enumerate(top_results):
            model = res["_model"]
            X_te = res["_X_te"]
            y_te = res["_y_te"]
            y_pred = model.predict(X_te)
            residuals = y_te.values - y_pred

            row = idx // cols + 1
            col = idx % cols + 1
            fig.add_trace(
                go.Scatter(
                    x=y_pred, y=residuals, mode="markers",
                    marker=dict(color=COLOR_SEQUENCE[idx], opacity=0.65, size=5),
                    name=res["Model"],
                    hovertemplate="Predicted: %{x:.3f}<br>Residual: %{y:.3f}<extra></extra>",
                ),
                row=row, col=col,
            )
            fig.add_hline(y=0, line_dash="dash", line_color="#374151", row=row, col=col)

        fig.update_layout(
            title="Residual Plots — Predicted vs Residuals",
            height=400 * rows,
            showlegend=False,
            **PLOTLY_DARK_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)


def plot_results_bar(metrics_df: pd.DataFrame, primary_metric: str, task: str):
    fig = go.Figure()
    metric_cols = [c for c in metrics_df.columns if c != "Model"]

    for i, col in enumerate(metric_cols):
        fig.add_trace(go.Bar(
            name=col,
            x=metrics_df["Model"],
            y=metrics_df[col],
            marker_color=COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)],
            hovertemplate=f"<b>%{{x}}</b><br>{col}: %{{y:.4f}}<extra></extra>",
        ))

    fig.update_layout(
        barmode="group",
        title="Model Comparison — All Metrics",
        height=420,
        **PLOTLY_DARK_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
for key, default in {
    "df": None,
    "X": None,
    "y": None,
    "le_target": None,
    "feature_names": [],
    "task": None,
    "results": None,
    "trained": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-family:Syne,sans-serif;font-weight:800;font-size:1.3rem;color:#a78bfa;margin-bottom:1.5rem;">⚡ AutoML Controls</div>', unsafe_allow_html=True)

    st.markdown("##### Data Ingestion")
    uploaded_files = st.file_uploader(
        "Upload CSV / Parquet",
        type=["csv", "parquet"],
        accept_multiple_files=True,
        help="You may upload multiple files — they'll be merged automatically.",
    )
    url_input = st.text_input("…or paste a URL", placeholder="https://…/data.csv")

    st.markdown("---")
    st.markdown("##### Feature Engineering")
    encoding_method = st.radio(
        "Categorical Encoding",
        ["Label Encoding", "One-Hot Encoding"],
        horizontal=False,
    )
    test_split = st.slider(
        "Test Split Ratio",
        min_value=0.10, max_value=0.40,
        value=0.20, step=0.05,
        format="%.2f",
    )

    st.markdown("---")
    st.markdown("##### Algorithm Selection")
    all_algos = ["Logistic Regression", "Linear Regression", "Random Forest",
                 "SVM", "XGBoost", "LightGBM"]
    auto_select = st.checkbox("Auto-Select All", value=True)
    if auto_select:
        selected_algos = all_algos
        st.caption("All algorithms selected ✓")
    else:
        selected_algos = st.multiselect(
            "Choose algorithms",
            options=all_algos,
            default=["Random Forest"],
        )
        if not XGBOOST_AVAILABLE and "XGBoost" in selected_algos:
            st.warning("XGBoost not installed.")
        if not LGBM_AVAILABLE and "LightGBM" in selected_algos:
            st.warning("LightGBM not installed.")

    st.markdown("---")
    run_btn = st.button("🚀  Run AutoML Pipeline", use_container_width=True)


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown('<div class="main-header">AutoML Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">End-to-end ML pipeline · Ingest · Clean · Train · Evaluate · Visualize</div>', unsafe_allow_html=True)

# ── Step 1: Data Ingestion ────────────────────
section_tag("STEP 01 — DATA INGESTION")
if uploaded_files or url_input.strip():
    df = ingest_data(uploaded_files, url_input)
    if df is not None:
        st.session_state.df = df
        c1, c2, c3 = st.columns(3)
        with c1: card("Rows", f"{df.shape[0]:,}")
        with c2: card("Columns", f"{df.shape[1]}")
        with c3: card("Missing Values", f"{df.isnull().sum().sum():,}")
        with st.expander("📋  Preview Dataset", expanded=False):
            st.dataframe(df.head(50), use_container_width=True)
else:
    st.markdown(
        '<div class="info-box">Upload a CSV/Parquet file or enter a dataset URL in the sidebar to begin.</div>',
        unsafe_allow_html=True,
    )

df = st.session_state.df
if df is not None:
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Step 2: Target Selection & Cleaning ──────
    section_tag("STEP 02 — TARGET & AUTO-CLEANING")
    target_col = st.selectbox(
        "Select Target Variable",
        options=df.columns.tolist(),
        index=len(df.columns) - 1,
    )

    if run_btn:
        if not selected_algos:
            st.error("Please select at least one algorithm.")
        else:
            with st.spinner("Auto-cleaning data…"):
                X, y, le_target, feature_names, task = auto_clean(df, target_col, encoding_method)
                st.session_state.X = X
                st.session_state.y = y
                st.session_state.le_target = le_target
                st.session_state.feature_names = feature_names
                st.session_state.task = task

            task_label = "Classification" if task == "classification" else "Regression"
            task_cls = "task-classification" if task == "classification" else "task-regression"
            st.markdown(
                f'Auto-detected task: <span class="task-pill {task_cls}">{task_label}</span>',
                unsafe_allow_html=True,
            )

            # ── Step 3+4: Train ──────────────────────
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            section_tag("STEP 03+04 — TRAINING & EVALUATION")
            models = get_models(task, selected_algos)
            if not models:
                st.error("No matching models for the selected task / algorithms.")
            else:
                results = run_training(models, X, y, test_split, task)
                st.session_state.results = results
                st.session_state.trained = True

    # ── Results ──────────────────────────────────
    if st.session_state.trained and st.session_state.results:
        results = st.session_state.results
        task = st.session_state.task
        X = st.session_state.X
        feature_names = st.session_state.feature_names

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        section_tag("RESULTS — MODEL COMPARISON")

        if task == "classification":
            display_cols = ["Model", "Accuracy", "F1 Score", "ROC-AUC"]
            primary = "Accuracy"
        else:
            display_cols = ["Model", "R²", "RMSE", "MAE"]
            primary = "R²"

        metrics_df = pd.DataFrame([
            {k: v for k, v in r.items() if not k.startswith("_")}
            for r in results
        ])

        ascending = primary in ("RMSE", "MAE")
        metrics_df = metrics_df.sort_values(primary, ascending=ascending).reset_index(drop=True)

        best = metrics_df.iloc[0] if ascending else metrics_df.iloc[-1]
        best_name = best["Model"]

        st.markdown(
            f'<div class="best-model-highlight">'
            f'<div class="best-model-title">🏆 Best Performing Model</div>'
            f'<div class="best-model-name">{best_name}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Styled metrics table
        fmt_dict = {c: "{:.4f}" for c in display_cols if c != "Model"}
        st.dataframe(
            metrics_df[display_cols].style.format(fmt_dict).background_gradient(
                subset=[primary], cmap="RdYlGn" if primary != "RMSE" else "RdYlGn_r"
            ),
            use_container_width=True,
        )

        # Bar comparison
        plot_results_bar(metrics_df[display_cols], primary, task)

        # ── Step 5: Visualizations ───────────────
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        section_tag("STEP 05 — VISUALIZATIONS")

        tab1, tab2, tab3 = st.tabs([
            "🔥  Correlation Heatmap",
            "📊  Feature Importance",
            "📈  ROC / Residuals",
        ])

        with tab1:
            plot_correlation_heatmap(df, target_col)

        with tab2:
            # Find best model object
            best_res = next((r for r in results if r["Model"] == best_name), results[0])
            plot_feature_importance(best_res["_model"], feature_names, best_name)

        with tab3:
            plot_roc_curves(results, task)

        # ── Extra: Class/value distribution ─────
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        section_tag("TARGET DISTRIBUTION")
        y_series = st.session_state.y
        if task == "classification":
            val_counts = y_series.value_counts().reset_index()
            val_counts.columns = ["Class", "Count"]
            le = st.session_state.le_target
            if le is not None:
                val_counts["Class"] = le.inverse_transform(val_counts["Class"].astype(int))
            fig = px.bar(val_counts, x="Class", y="Count", color="Count",
                         color_continuous_scale=["#4c1d95", "#7c3aed", "#00d4ff"])
        else:
            fig = px.histogram(y_series, nbins=40, color_discrete_sequence=["#7c3aed"])
            fig.update_layout(xaxis_title=target_col, yaxis_title="Count")

        fig.update_layout(title="Target Variable Distribution", height=360, **PLOTLY_DARK_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:3rem;padding:1.5rem;
            border-top:1px solid #1e1e30;font-family:'JetBrains Mono',monospace;
            font-size:0.72rem;color:#374151;">
    ⚡ AutoML Dashboard — scikit-learn · XGBoost · LightGBM · Streamlit
</div>
""", unsafe_allow_html=True)
