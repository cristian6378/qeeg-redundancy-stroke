"""The label-agnostic redundancy chain: correlation screening by priority, then iterative VIF."""
import numpy as np
import pandas as pd

import config as C
import selection as S


def _frame(n=500, seed=1):
    rng = np.random.default_rng(seed)
    base = {f: rng.normal(size=n) for f in C.FEATURES_17}
    base["delta_alpha"] = base["delta"] + 0.05 * rng.normal(size=n)   # ratio tracks its band power
    base["kurtosis"] = base["skewness"] + 0.05 * rng.normal(size=n)
    return pd.DataFrame(base)


def test_stage1_keeps_higher_priority_member():
    keep, dropped = S.stage1_correlation(_frame())
    assert "delta" in keep and "delta_alpha" in dropped          # Study-1 nine member kept
    assert "skewness" in keep and "kurtosis" in dropped


def test_stage2_removes_exact_collinearity():
    df = _frame()
    df["beta"] = df["theta"] + df["alpha"] + 1e-3 * np.random.default_rng(2).normal(size=len(df))
    keep, removed, _ = S.stage2_vif(df[["theta", "alpha", "beta", "skewness"]])
    assert len(removed) >= 1
    assert S.vif(df[keep]).max() < C.VIF_THRESHOLD


def test_fit_redundancy_uses_no_labels_and_is_deterministic():
    df = _frame()
    k1, r1 = S.fit_redundancy(df)
    k2, r2 = S.fit_redundancy(df.copy())
    assert k1 == k2 and r1 == r2
    assert r1["n_final"] == len(k1) <= 17
