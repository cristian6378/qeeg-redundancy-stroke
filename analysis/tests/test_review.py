"""Units of the review-stage analyses (task-matched windows, hierarchy, benchmark, uncertainty)."""
import numpy as np
import pandas as pd
import pytest

import common as K
import A3_hierarchical_synthetic as A3
import B3_taskmatched as B3
import B4_hierarchy as B4
import B6_uncertainty as B6


def test_windows_stay_inside_their_phase():
    spec = B3.window_spec(onset=2.0, mi_len=4.0)
    bounds = {"instruction": (0, 2), "imagery": (2, 6), "imagery_late": (2, 6), "rest": (6, 8), "full8": (0, 8)}
    for phase, L, start in spec:
        lo, hi = bounds[phase]
        assert lo <= start and start + L <= hi, (phase, L, start)
    assert ("imagery", 4, 2.0) in spec and ("rest", 2, 6.0) in spec


def _trials(n_sub=12, n_tr=40, seed=0, flag=False):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sub):
        mu = rng.normal(0, 1, 17)
        for t in range(n_tr):
            f = bool(flag and rng.random() < 0.1)
            for phase, L, start in B3.window_spec(2.0, 4.0):
                v = np.exp(mu + rng.normal(0, 0.7, 17))
                rows.append({"subject": s, "trial": t, "hand": 1 + t % 2, "phase": phase, "length_s": L,
                             "start_s": start, "qc_flag": f, **dict(zip(K.F17, v))})
    return pd.DataFrame(rows)


def test_icc_table_recovers_between_subject_structure():
    tab = B3.icc_table(_trials(), n_boot=10)
    a = tab[tab.hand == "all"]
    # simulated ICC = 1 / (1 + 0.7^2) = 0.67 for the log-transformed features (12 subjects: wide tolerance)
    logged = a[a.feature.isin(K.LOG_FEATURES)]
    assert logged.icc1.median() == pytest.approx(0.67, abs=0.15)
    ok = a.seconds_lo.notna()
    assert (a.seconds_lo[ok] <= a.seconds_hi[ok]).all() and (a.seconds_lo[ok] > 0).all()
    assert {"left", "right"} <= set(tab.hand)


def test_hierarchy_levels_have_the_right_size():
    T = _trials()
    V = T[(T.phase == "imagery") & (T.length_s == 4)].assign(kind="length")
    summ, sim, freq = B4.run(V, n_boot=5, write=False)
    n = summ.set_index("level").n_rows
    assert n["between"] == 12 and n["pooled"] == n["within"] == 12 * 40
    assert freq.values.min() >= 0 and freq.values.max() <= 1


def test_benchmark_collinearity_increases_with_rho():
    vifs = [A3.max_vif(A3.generate(0.6, rho, n_subjects=60, n_trials=5)) for rho in (0.3, 0.6, 0.9)]
    assert vifs[0] < vifs[1] < vifs[2] and vifs[2] > 10
    df = A3.generate(0.6, 0.6, n_subjects=40, n_trials=3)
    assert df.groupby("subject_id").label.nunique().max() == 1     # one class per subject
    assert df.subject_id.nunique() == 40


def test_paired_differences_are_centred_on_the_point_estimate():
    T = B3._prep(_trials(flag=True)).drop(columns=[])
    out = B6.taskmatched(T, None, n_boot=50)
    d = out[out.quantity == "second_minus_first"]
    assert (d.lo <= d.estimate + 1e-9).all() and (d.estimate <= d.hi + 1e-9).all()
    # no real drift was simulated: most intervals should include zero
    assert (~d.excludes_zero.astype(bool)).mean() >= 0.8
