"""Sanity checks of the qEEG descriptors."""
import numpy as np
import pytest

import config as C
import features as F

FS = C.FS_RAW


def test_band_power_peaks_in_the_right_band():
    t = np.arange(2 * FS) / FS
    x = np.vstack([np.sin(2 * np.pi * 10 * t), np.sin(2 * np.pi * 2 * t)])   # alpha, delta
    bp = F.band_powers(x, FS)
    assert bp["alpha"][0] > 100 * bp["delta"][0]
    assert bp["delta"][1] > 100 * bp["alpha"][1]


def test_homologous_pairs_are_matched_by_name():
    names = ["FP1", "FP2", "C3", "CZ", "C4", "O1", "O2"]
    pairs = F.homologous_index_pairs(names)
    assert (2, 4) in pairs and (0, 1) in pairs and (5, 6) in pairs
    assert all(names[a] != "CZ" and names[b] != "CZ" for a, b in pairs)


def test_pdbsi_is_zero_for_symmetric_power_and_bounded():
    pairs = [(0, 1), (2, 3)]
    assert F.pdbsi(np.array([1.0, 1.0, 2.0, 2.0]), pairs) == pytest.approx(0.0)
    assert 0 <= F.pdbsi(np.array([1.0, 3.0, 5.0, 0.5]), pairs) <= 1


def test_trial_features_returns_all_17():
    rng = np.random.default_rng(0)
    names = ["F3", "F4", "C3", "C4", "P3", "P4"]
    x = rng.normal(size=(len(names), 2 * FS))
    out = F.trial_features(x, FS, F.homologous_index_pairs(names))
    assert set(C.FEATURES_17) <= set(out)
    assert all(np.isfinite(out[f]) for f in C.FEATURES_17)
