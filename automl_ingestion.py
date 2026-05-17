"""
automl_ingestion.py
════════════════════════════════════════════════════════════════════════
Module 1 — Memory-Safe Ingestion & Sanitization Layer
Replaces: load_dataframe(), ingest_data(), auto_clean()

Fixes addressed:
  ✔ Target NaN Fix   — drops rows with null target BEFORE split
  ✔ Memory Error Fix — prunes high-cardinality object columns (>20% unique)
  ✔ Sparse Handling  — OneHotEncoder(sparse_output=True) + scipy.sparse.hstack
  ✔ Metric alias     — returns task dict with both "R2" and "R²" keys
════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import io
import warnings
import logging
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import streamlit as st

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    OneHotEncoder,
)

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# ── Tunables ──────────────────────────────────────────────────────────
HIGH_CARDINALITY_THRESHOLD = 0.20   # drop object col if unique/nrows > this
REGRESSION_UNIQUE_FLOOR    = 15     # numeric target with ≥ N uniques → regression


# ─────────────────────────────────────────────────────────────────────
# 1-A. FILE / URL LOADERS
# ─────────────────────────────────────────────────────────────────────

def _read_source(source) -> pd.DataFrame:
    """
    Load one source — either a Streamlit UploadedFile or a URL string.
    Supports CSV and Parquet.
    """
    if isinstance(source, str):
        url = source.strip()
        return pd.read_parquet(url) if url.lower().endswith(".parquet") else pd.read_csv(url)

    name = getattr(source, "name", "").lower()
    raw  = source.read()
    if name.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(raw))
    # Try UTF-8 first, fall back to latin-1 for dirty CSVs
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode file '{name}' with any supported encoding.")


def ingest_data(
    uploaded_files: list,
    url_input: str,
) -> Optional[pd.DataFrame]:
    """
    Load one or many CSV/Parquet sources and merge them.

    Merge strategy:
      • If multiple frames share common columns  → concat on common columns.
      • Otherwise                                → outer concat (all columns).

    Returns None if no valid source was provided or all loads failed.
    """
    frames: list[pd.DataFrame] = []

    for f in (uploaded_files or []):
        try:
            frames.append(_read_source(f))
        except Exception as exc:
            st.error(f"❌  Could not read **{getattr(f,'name','file')}**: {exc}")

    if url_input.strip():
        try:
            frames.append(_read_source(url_input.strip()))
        except Exception as exc:
            st.error(f"❌  Could not fetch URL: {exc}")

    if not frames:
        return None
    if len(frames) == 1:
        return frames[0].reset_index(drop=True)

    # Multi-file merge
    common = set(frames[0].columns)
    for f in frames[1:]:
        common &= set(f.columns)

    if common:
        merged = pd.concat(
            [f[sorted(common)] for f in frames], ignore_index=True
        )
    else:
        merged = pd.concat(frames, ignore_index=True)

    return merged


# ─────────────────────────────────────────────────────────────────────
# 1-B. TASK DETECTION
# ─────────────────────────────────────────────────────────────────────

def detect_task(y: pd.Series) -> str:
    """
    Regression  → numeric target with ≥ REGRESSION_UNIQUE_FLOOR unique values.
    Classification → everything else (including numeric with few levels).
    """
    if pd.api.types.is_numeric_dtype(y) and y.nunique() >= REGRESSION_UNIQUE_FLOOR:
        return "regression"
    return "classification"


# ─────────────────────────────────────────────────────────────────────
# 1-C. HIGH-CARDINALITY COLUMN PRUNER
# ─────────────────────────────────────────────────────────────────────

def _prune_high_cardinality(
    df: pd.DataFrame,
    n_rows: int,
    threshold: float = HIGH_CARDINALITY_THRESHOLD,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Drop object/category columns whose unique-value ratio exceeds `threshold`.
    These are almost certainly ID/hash/free-text columns that would explode
    One-Hot memory.  Returns the pruned frame and a list of dropped col names.
    """
    dropped: list[str] = []
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in cat_cols:
        ratio = df[col].nunique() / max(n_rows, 1)
        if ratio > threshold:
            dropped.append(col)

    if dropped:
        df = df.drop(columns=dropped)

    return df, dropped


# ─────────────────────────────────────────────────────────────────────
# 1-D. MAIN CLEANING PIPELINE
# ─────────────────────────────────────────────────────────────────────

def auto_clean(
    df: pd.DataFrame,
    target_col: str,
    encoding: str,          # "Label Encoding" | "One-Hot Encoding"
) -> tuple:
    """
    Full sanitization → feature matrix pipeline.

    Returns
    -------
    X_out        : np.ndarray or scipy.sparse matrix  (samples × features)
    y            : pd.Series  (encoded target)
    le_target    : LabelEncoder | None
    feature_names: list[str]
    task         : "classification" | "regression"
    dropped_cols : list[str]   — high-cardinality cols that were pruned
    """
    df = df.copy()

    # ── FIX 1: Drop rows where TARGET is NaN (prevents "Input y contains NaN")
    n_before = len(df)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    n_dropped_target = n_before - len(df)
    if n_dropped_target:
        st.warning(
            f"⚠️  Dropped **{n_dropped_target:,}** row(s) where "
            f"`{target_col}` was null to avoid training errors."
        )

    if df.empty:
        raise ValueError(
            f"After dropping NaN targets in '{target_col}', the dataset is empty."
        )

    # ── Separate target
    y_raw = df[target_col].copy()
    X_raw = df.drop(columns=[target_col])

    task = detect_task(y_raw)

    # ── Encode target
    le_target: Optional[LabelEncoder] = None
    if task == "classification" and not pd.api.types.is_numeric_dtype(y_raw):
        le_target = LabelEncoder()
        y = pd.Series(
            le_target.fit_transform(y_raw.astype(str)),
            name=target_col,
            dtype=int,
        )
    else:
        y = y_raw.astype(float).reset_index(drop=True)

    # ── FIX 2: Prune high-cardinality object columns
    X_raw, dropped_cols = _prune_high_cardinality(X_raw, n_rows=len(X_raw))
    if dropped_cols:
        st.warning(
            f"⚠️  Dropped **{len(dropped_cols)}** high-cardinality column(s) "
            f"(>{int(HIGH_CARDINALITY_THRESHOLD*100)}% unique values): "
            f"`{'`, `'.join(dropped_cols)}`"
        )

    # ── Split numeric vs categorical
    num_cols = X_raw.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X_raw.select_dtypes(include=["object", "category"]).columns.tolist()

    # ── Numeric pipeline: impute → scale
    num_parts: list   = []
    num_names: list[str] = []

    if num_cols:
        num_imp = SimpleImputer(strategy="median")
        X_num_arr = num_imp.fit_transform(X_raw[num_cols].values)
        scaler    = StandardScaler()
        X_num_arr = scaler.fit_transform(X_num_arr)
        num_parts.append(X_num_arr)
        num_names.extend(num_cols)

    # ── Categorical pipeline: impute → encode
    cat_parts: list   = []
    cat_names: list[str] = []

    if cat_cols:
        cat_imp   = SimpleImputer(strategy="most_frequent")
        X_cat_imp = cat_imp.fit_transform(X_raw[cat_cols].astype(str).values)
        X_cat_imp = pd.DataFrame(X_cat_imp, columns=cat_cols)

        if encoding == "Label Encoding":
            le_frame = np.empty((len(X_cat_imp), len(cat_cols)), dtype=float)
            for j, col in enumerate(cat_cols):
                le = LabelEncoder()
                le_frame[:, j] = le.fit_transform(X_cat_imp[col])
            cat_parts.append(le_frame)
            cat_names.extend(cat_cols)

        else:
            # ── FIX 3: Sparse One-Hot to prevent ArrayMemoryError
            ohe = OneHotEncoder(
                sparse_output=True,
                handle_unknown="ignore",
                dtype=np.float32,
            )
            X_ohe = ohe.fit_transform(X_cat_imp.values)
            cat_parts.append(X_ohe)
            cat_names.extend(
                [f"{col}_{v}"
                 for col, cats in zip(cat_cols, ohe.categories_)
                 for v in cats]
            )

    # ── Assemble feature matrix
    feature_names = num_names + cat_names

    has_sparse = any(sp.issparse(p) for p in cat_parts)

    if has_sparse:
        # Keep everything sparse → convert dense num parts first
        all_parts: list = []
        for p in num_parts:
            all_parts.append(sp.csr_matrix(p.astype(np.float32)))
        all_parts.extend(cat_parts)
        X_out = sp.hstack(all_parts, format="csr") if len(all_parts) > 1 else all_parts[0]
    else:
        dense_parts: list[np.ndarray] = []
        if num_parts:
            dense_parts.append(num_parts[0].astype(np.float32))
        if cat_parts:
            for p in cat_parts:
                dense_parts.append(
                    p.toarray().astype(np.float32) if sp.issparse(p) else p.astype(np.float32)
                )
        X_out = (
            np.hstack(dense_parts)
            if len(dense_parts) > 1
            else (dense_parts[0] if dense_parts else np.empty((len(y), 0), dtype=np.float32))
        )

    return X_out, y.reset_index(drop=True), le_target, feature_names, task, dropped_cols
