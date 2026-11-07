"""Label-agnostic redundancy reduction (Study 1 rule set, automated so it can run inside folds).

Stage 1: for every feature pair with |Pearson r| >= CORR_THRESHOLD, drop the member that is
         lower in PRIORITY (Study 1 kept base band powers, pDBSI and skewness over ratios and
         kurtosis "on grounds of interpretability and domain coverage").
Stage 2: iteratively drop the feature with the largest VIF until every VIF < VIF_THRESHOLD.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C

PRIORITY = C.FEATURES_9_STUDY1 + [f for f in C.FEATURES_17 if f not in C.FEATURES_9_STUDY1]


def vif(df: pd.DataFrame) -> pd.Series:
    """VIF_j = 1 / (1 - R^2_j), R^2 from OLS of standardized feature j on the others."""
    Z = (df - df.mean()) / (df.std(ddof=0) + 1e-12)
    Z = Z.values
    out = {}
    for j, name in enumerate(df.columns):
        y = Z[:, j]
        Xo = np.delete(Z, j, axis=1)
        Xo = np.column_stack([np.ones(len(Xo)), Xo])
        beta, *_ = np.linalg.lstsq(Xo, y, rcond=None)
        resid = y - Xo @ beta
        r2 = 1 - resid.var() / (y.var() + 1e-12)
        out[name] = np.inf if r2 >= 1 - 1e-12 else 1.0 / (1.0 - r2)
    return pd.Series(out)


def stage1_correlation(df: pd.DataFrame, thr=C.CORR_THRESHOLD, method="pearson"):
    r = df.corr(method=method).abs()
    cols = list(df.columns)
    rank = {f: PRIORITY.index(f) if f in PRIORITY else len(PRIORITY) for f in cols}
    pairs = [(r.iloc[i, j], cols[i], cols[j]) for i in range(len(cols)) for j in range(i + 1, len(cols))
             if r.iloc[i, j] >= thr]
    dropped = set()
    for _, a, b in sorted(pairs, reverse=True):
        if a in dropped or b in dropped:
            continue
        dropped.add(a if rank[a] > rank[b] else b)
    return [c for c in cols if c not in dropped], sorted(dropped, key=lambda f: rank[f])


def stage2_vif(df: pd.DataFrame, thr=C.VIF_THRESHOLD):
    cols = list(df.columns)
    removed, history = [], []
    while len(cols) > 2:
        v = vif(df[cols])
        history.append(v.round(2).to_dict())
        if v.max() < thr:
            break
        worst = v.idxmax()
        removed.append((worst, float(v.max())))
        cols.remove(worst)
    return cols, removed, history


def fit_redundancy(df: pd.DataFrame, corr_thr=C.CORR_THRESHOLD, vif_thr=C.VIF_THRESHOLD):
    """Return (retained, report) for a training-fold feature table (no labels used)."""
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    s1_keep, s1_drop = stage1_correlation(df, thr=corr_thr)
    s2_keep, s2_drop, _ = stage2_vif(df[s1_keep], thr=vif_thr)
    return s2_keep, {"stage1_dropped": s1_drop, "stage2_dropped": s2_drop,
                     "n_after_stage1": len(s1_keep), "n_final": len(s2_keep)}
