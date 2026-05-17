"""
automl_models.py
════════════════════════════════════════════════════════════════════════
Module 2 — Expanded Model Registry & Isolated Training Engine
Replaces: get_models(), run_training(), train_evaluate_*()

Fixes addressed:
  ✔ Metric alias    — evaluation dict carries BOTH "R2" and "R²" keys
  ✔ Isolated fit()  — per-model try/except; failure skips model gracefully
  ✔ Algorithm scope — 8 classifiers, 9 regressors (CatBoost included)
════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import time
import traceback
import warnings
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
    Lasso,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    roc_curve,
    auc,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

# ── Optional heavy dependencies ───────────────────────────────────────

def _try_import(module: str, cls: str):
    try:
        import importlib
        m = importlib.import_module(module)
        return getattr(m, cls)
    except (ImportError, AttributeError):
        return None

XGBClassifier      = _try_import("xgboost",  "XGBClassifier")
XGBRegressor       = _try_import("xgboost",  "XGBRegressor")
LGBMClassifier     = _try_import("lightgbm", "LGBMClassifier")
LGBMRegressor      = _try_import("lightgbm", "LGBMRegressor")
CatBoostClassifier = _try_import("catboost", "CatBoostClassifier")
CatBoostRegressor  = _try_import("catboost", "CatBoostRegressor")

XGBOOST_OK  = XGBClassifier  is not None
LGBM_OK     = LGBMClassifier is not None
CATBOOST_OK = CatBoostClassifier is not None


# ─────────────────────────────────────────────────────────────────────
# 2-A. MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────────

#: Maps human-readable algo name → whether the library is available
ALGO_AVAILABILITY: dict[str, bool] = {
    # ── classifiers
    "Logistic Regression":    True,
    "Random Forest":          True,
    "Extra Trees":            True,
    "Gradient Boosting":      True,
    "XGBoost":                XGBOOST_OK,
    "LightGBM":               LGBM_OK,
    "CatBoost":               CATBOOST_OK,
    "SVC":                    True,
    # ── regressors
    "Linear Regression":      True,
    "Ridge":                  True,
    "Lasso":                  True,
    "Random Forest Regressor":         True,
    "Extra Trees Regressor":           True,
    "Gradient Boosting Regressor":     True,
    "XGBoost Regressor":               XGBOOST_OK,
    "LightGBM Regressor":              LGBM_OK,
    "CatBoost Regressor":              CATBOOST_OK,
}

#: Human-readable names visible in sidebar, keyed by task
ALGO_NAMES_BY_TASK: dict[str, list[str]] = {
    "classification": [
        "Logistic Regression",
        "Random Forest",
        "Extra Trees",
        "Gradient Boosting",
        "XGBoost",
        "LightGBM",
        "CatBoost",
        "SVC",
    ],
    "regression": [
        "Linear Regression",
        "Ridge",
        "Lasso",
        "Random Forest Regressor",
        "Extra Trees Regressor",
        "Gradient Boosting Regressor",
        "XGBoost Regressor",
        "LightGBM Regressor",
        "CatBoost Regressor",
    ],
}

# Unified list for the sidebar (shown before task is detected)
ALL_ALGO_NAMES: list[str] = sorted(set(
    ALGO_NAMES_BY_TASK["classification"] + ALGO_NAMES_BY_TASK["regression"]
))


def _build_classifier(name: str) -> Any | None:
    """Instantiate one classifier. Returns None if library unavailable."""
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    if name == "Extra Trees":
        return ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(n_estimators=150, random_state=42)
    if name == "XGBoost" and XGBOOST_OK:
        return XGBClassifier(
            n_estimators=200, random_state=42,
            eval_metric="logloss", verbosity=0,
            use_label_encoder=False,
        )
    if name == "LightGBM" and LGBM_OK:
        return LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
    if name == "CatBoost" and CATBOOST_OK:
        return CatBoostClassifier(
            iterations=200, random_state=42,
            verbose=0, allow_writing_files=False,
        )
    if name == "SVC":
        return SVC(probability=True, random_state=42)
    return None


def _build_regressor(name: str) -> Any | None:
    """Instantiate one regressor. Returns None if library unavailable."""
    if name == "Linear Regression":
        return LinearRegression(n_jobs=-1)
    if name == "Ridge":
        return Ridge(alpha=1.0, random_state=42)
    if name == "Lasso":
        return Lasso(alpha=0.01, max_iter=5000, random_state=42)
    if name == "Random Forest Regressor":
        return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    if name == "Extra Trees Regressor":
        return ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    if name == "Gradient Boosting Regressor":
        return GradientBoostingRegressor(n_estimators=150, random_state=42)
    if name == "XGBoost Regressor" and XGBOOST_OK:
        return XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
    if name == "LightGBM Regressor" and LGBM_OK:
        return LGBMRegressor(n_estimators=200, random_state=42, verbose=-1)
    if name == "CatBoost Regressor" and CATBOOST_OK:
        return CatBoostRegressor(
            iterations=200, random_state=42,
            verbose=0, allow_writing_files=False,
        )
    return None


def get_models(task: str, selected_names: list[str]) -> dict[str, Any]:
    """
    Build and return {name: estimator} for the requested task and selection.
    Algorithms whose library is unavailable or whose name doesn't match the
    task's registry are silently skipped.
    """
    valid_names = set(ALGO_NAMES_BY_TASK[task])
    models: dict[str, Any] = {}

    for name in selected_names:
        if name not in valid_names:
            continue
        estimator = (
            _build_classifier(name)
            if task == "classification"
            else _build_regressor(name)
        )
        if estimator is None:
            st.warning(
                f"⚠️  **{name}** is not installed in this environment — skipping."
            )
        else:
            models[name] = estimator

    return models


# ─────────────────────────────────────────────────────────────────────
# 2-B. ISOLATED EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────────

def _roc_auc_safe(model, X_te, y_te) -> float:
    """ROC-AUC for binary and multiclass (OVR/weighted). Returns NaN on failure."""
    try:
        if not hasattr(model, "predict_proba"):
            return float("nan")
        y_prob = model.predict_proba(X_te)
        n_cls  = len(np.unique(y_te))
        if n_cls == 2:
            return roc_auc_score(y_te, y_prob[:, 1])
        return roc_auc_score(y_te, y_prob, multi_class="ovr", average="weighted")
    except Exception:
        return float("nan")


def _eval_classification(model, X_tr, X_te, y_tr, y_te, name: str) -> dict:
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    acc  = accuracy_score(y_te, y_pred)
    f1   = f1_score(y_te, y_pred, average="weighted", zero_division=0)
    rauc = _roc_auc_safe(model, X_te, y_te)

    return {
        "Model":    name,
        "Accuracy": acc,
        "F1 Score": f1,
        "ROC-AUC":  rauc,
        # ── internal objects (excluded from display tables)
        "_model": model,
        "_y_pred": y_pred,
    }


def _eval_regression(model, X_tr, X_te, y_tr, y_te, name: str) -> dict:
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    r2   = r2_score(y_te, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
    mae  = float(mean_absolute_error(y_te, y_pred))

    return {
        "Model": name,
        # ── FIX: both aliases present to prevent KeyError on either lookup
        "R²":   r2,
        "R2":   r2,
        "RMSE": rmse,
        "MAE":  mae,
        "_model":  model,
        "_y_pred": y_pred,
    }


# ─────────────────────────────────────────────────────────────────────
# 2-C. TRAINING ORCHESTRATOR (isolated per-model try/except)
# ─────────────────────────────────────────────────────────────────────

def run_training(
    models:    dict[str, Any],
    X:         Any,                  # ndarray or sparse matrix
    y:         pd.Series,
    test_size: float,
    task:      str,
) -> list[dict]:
    """
    Train every model in `models` sequentially with an isolated try/except
    per .fit() call.  A failing model emits st.error() and is skipped;
    all remaining models continue training normally.

    Returns a list of result dicts (one per successfully trained model).
    """
    # Convert sparse to CSR for slicing compatibility with all sklearn estimators
    if sp.issparse(X):
        X = X.tocsr()

    y_arr = y.values

    # Stratify only for classification
    stratify = y_arr if task == "classification" else None

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_arr,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )

    results: list[dict] = []
    total   = len(models)
    prog    = st.progress(0, text="Initialising training pipeline…")

    for i, (name, model) in enumerate(models.items()):
        prog.progress(i / total, text=f"Training  **{name}**…")
        t0 = time.perf_counter()

        try:
            if task == "classification":
                res = _eval_classification(model, X_tr, X_te, y_tr, y_te, name)
            else:
                res = _eval_regression(model, X_tr, X_te, y_tr, y_te, name)

            elapsed = time.perf_counter() - t0
            res["_elapsed_s"] = round(elapsed, 2)
            res["_X_tr"] = X_tr
            res["_X_te"] = X_te
            res["_y_tr"] = y_tr
            res["_y_te"] = y_te
            results.append(res)

        except MemoryError:
            st.error(
                f"⛔  **{name}** — MemoryError: dataset too large for this algorithm. "
                f"Try Label Encoding or reduce features."
            )
        except Exception as exc:
            st.error(
                f"⛔  **{name}** failed with `{type(exc).__name__}: {exc}`. "
                f"Skipping and continuing with remaining models."
            )
            # Full traceback to Streamlit expander for debugging
            with st.expander(f"🔍  Debug traceback for {name}", expanded=False):
                st.code(traceback.format_exc(), language="python")

        prog.progress((i + 1) / total, text=f"✅  {name} done")

    prog.progress(1.0, text=f"🏁  Training complete — {len(results)}/{total} models succeeded.")
    return results


# ─────────────────────────────────────────────────────────────────────
# 2-D. METRICS TABLE BUILDER
# ─────────────────────────────────────────────────────────────────────

def build_metrics_df(results: list[dict], task: str) -> pd.DataFrame:
    """
    Construct a clean display DataFrame from raw result dicts.
    Private keys (starting with '_') are excluded.
    Returns DataFrame sorted by primary metric.
    """
    if task == "classification":
        cols    = ["Model", "Accuracy", "F1 Score", "ROC-AUC"]
        primary = "Accuracy"
    else:
        # Use "R²" for display; "R2" alias prevents KeyError internally
        cols    = ["Model", "R²", "RMSE", "MAE"]
        primary = "R²"

    rows = []
    for r in results:
        row = {}
        for k, v in r.items():
            if not k.startswith("_"):
                row[k] = v
        rows.append(row)

    df = pd.DataFrame(rows)

    # Ensure all expected columns exist (fill NaN for failed metrics)
    for c in cols:
        if c not in df.columns:
            df[c] = float("nan")

    ascending = primary in ("RMSE", "MAE")
    df = df.sort_values(primary, ascending=ascending).reset_index(drop=True)
    return df, cols, primary
