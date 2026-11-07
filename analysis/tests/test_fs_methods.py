"""Selection strategies recover planted structure and respect their definitions."""
import numpy as np
import pandas as pd
import pytest

import common as K
import fs_methods as SEL


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1)
    n = 600
    X = pd.DataFrame(rng.normal(size=(n, 17)), columns=K.F17)
    y = (X.alpha + X.pdbsi_delta + 0.5 * rng.normal(size=n) > 0).astype(int).values
    X["delta_alpha"] = X.alpha + 0.1 * rng.normal(size=n)        # redundant copy of alpha
    return X, y, np.repeat(np.arange(30), 20)


@pytest.mark.parametrize("method", ["mrmr", "enet", "stability", "rfe"])
def test_label_aware_methods_find_the_signal(data, method):
    X, y, g = data
    sel = SEL.select(method, X, y, g, "clf")
    assert "pdbsi_delta" in sel and ({"alpha", "delta_alpha"} & set(sel))
    assert len(sel) <= 8


def test_label_agnostic_methods_remove_the_duplicate(data):
    X, y, g = data
    for method in ("corr_vif", "cluster"):
        sel = SEL.select(method, X, y, g, "clf")
        assert not {"alpha", "delta_alpha"} <= set(sel)


def test_group_lasso_keeps_whole_families_and_is_empty_at_lambda_max(data):
    X, y, g = data
    sel = SEL.select("glasso", X, y, g, "clf")
    for fam in K.FAMILIES.values():
        assert set(fam) <= set(sel) or not set(fam) & set(sel)
    Xz = (X - X.mean()) / X.std()
    fam_idx = [[K.F17.index(f) for f in fs] for fs in K.FAMILIES.values()]
    W, _ = SEL.group_lasso_fit(Xz.values, y, fam_idx, lam=10.0, task="clf")
    assert np.allclose(W, 0)


def test_panel_pca_and_all(data):
    X, y, g = data
    assert SEL.select("all", X, y, g) == K.F17
    assert SEL.select("panel", X, y, g) == K.CLINICAL_PANEL
    assert SEL.select("pca", X, y, g) == "pca"
    p = SEL.PCAReference().fit(X.values)
    assert 1 <= p.n_components <= 17


def test_subset_cost_includes_band_powers_for_pdbsi():
    t = dict(K.COST_MS)
    assert K.subset_cost(["pdbsi_alpha"], t) == pytest.approx(t["pDBSI (4 bands)"] + t["band powers (+ratios)"])
    assert K.subset_cost(K.F17, t) == pytest.approx(sum(t.values()))
