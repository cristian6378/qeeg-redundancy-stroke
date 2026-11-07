"""Reliability, agreement and stability measures against published values and identities."""
import numpy as np
import pytest

import measures as M

# Shrout & Fleiss (1979), Table 2: 6 targets rated by 4 judges
SF = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]], float)


def test_icc_matches_shrout_fleiss_table():
    one = M.icc_oneway(SF)
    two = M.icc_twoway(SF)
    assert one["icc1"] == pytest.approx(0.17, abs=0.005)
    assert one["icck"] == pytest.approx(0.44, abs=0.005)
    assert two["icc2"] == pytest.approx(0.29, abs=0.005)
    assert two["icc3"] == pytest.approx(0.71, abs=0.005)
    assert one["lo"] < one["icc1"] < one["hi"]


def test_spearman_brown_inverts_icck():
    icc1 = M.icc_oneway(SF)["icc1"]
    k = M.spearman_brown_k(icc1, 0.44)
    assert k == pytest.approx(4, rel=0.05)


def test_ccc_identities():
    x = np.random.default_rng(0).normal(size=50)
    x = x - x.mean()
    assert M.ccc(x, x) == pytest.approx(1.0)
    assert M.ccc(x, x + 1.0) < 1.0           # a shift lowers concordance but not correlation
    assert M.ccc(x, -x) == pytest.approx(-1.0)


def test_nogueira_bounds():
    same = np.tile([1, 1, 0, 0, 1, 0], (20, 1))
    assert M.nogueira_stability(same)["phi"] == pytest.approx(1.0)
    rng = np.random.default_rng(1)
    rand = (rng.random((200, 30)) < 0.3).astype(int)
    r = M.nogueira_stability(rand)
    assert abs(r["phi"]) < 0.05 and r["lo"] < r["phi"] < r["hi"]


def test_jaccard_and_partial_spearman():
    assert M.mean_jaccard([["a", "b"], ["a", "b"], ["a", "c"]]) == pytest.approx((1 + 1 / 3 + 1 / 3) / 3)
    rng = np.random.default_rng(2)
    z = rng.normal(size=200)
    x, y = z + 0.1 * rng.normal(size=200), z + 0.1 * rng.normal(size=200)
    r_raw = M.partial_spearman(x, y, None)[0]
    r_part, _ = M.partial_spearman(x, y, [z])
    assert r_raw > 0.9 and abs(r_part) < 0.3   # the common cause explains the association
