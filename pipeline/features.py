"""The 17 Study-1 qEEG descriptors, computed per trial and aggregated over channels by the median.

Definitions follow the thesis notebooks (synthetic_df_feature_extraction.ipynb) so that the
real-EEG feature space is the same as the synthetic one. Homologous pairs are resolved by
channel NAME, which fixes the index-based pairing used in the earlier real-EEG checkpoint.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import hilbert, welch, csd
from scipy.stats import kurtosis, skew

import config as C

try:
    from antropy import sample_entropy as _ant_sampen, higuchi_fd as _ant_hfd
except Exception:  # pragma: no cover - fallback keeps the pipeline runnable without antropy
    _ant_sampen = _ant_hfd = None


# ----------------------------------------------------------------------------- primitives
def band_powers(x, fs):
    """Absolute power: sum of squared DFT magnitudes per band. x: [ch, n] -> {band: [ch]}"""
    n = x.shape[-1]
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    p = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    return {b: p[:, (freqs >= lo) & (freqs < hi)].sum(axis=-1) for b, (lo, hi) in C.BANDS.items()}


def _sampen_np(sig, m=2, r=None):
    sig = np.asarray(sig, dtype=np.float64)
    r = 0.2 * sig.std() if r is None else r
    n = len(sig)

    def _count(mm):  # template pairs j > i within tolerance r (Chebyshev distance)
        emb = np.lib.stride_tricks.sliding_window_view(sig, mm)[: n - m]
        cnt = 0
        for i in range(len(emb) - 1):
            d = np.max(np.abs(emb[i + 1:] - emb[i]), axis=1)
            cnt += int(np.sum(d <= r))
        return cnt

    b, a = _count(m), _count(m + 1)
    return -np.log(a / b) if a > 0 and b > 0 else np.nan


def sample_entropy(sig):
    if _ant_sampen is not None:
        return float(_ant_sampen(sig.astype(np.float64), order=2,
                                 tolerance=float(0.2 * np.std(sig))))
    return _sampen_np(sig)


def higuchi_fd(sig, kmax=10):
    if _ant_hfd is not None:
        return float(_ant_hfd(sig.astype(np.float64), kmax=kmax))
    n = len(sig)
    lk = []
    for k in range(1, kmax + 1):
        lm = []
        for m in range(k):
            idx = np.arange(m, n, k)
            if len(idx) < 2:
                continue
            lm.append(np.abs(np.diff(sig[idx])).sum() * (n - 1) / ((len(idx) - 1) * k) / k)
        lk.append(np.mean(lm))
    ks = np.arange(1, kmax + 1)
    return float(np.polyfit(np.log(1.0 / ks), np.log(lk), 1)[0])


def wpli_all_pairs(x):
    """Broadband wPLI from analytic signals, median over all unique channel pairs."""
    a = hilbert(x, axis=-1)
    iu = np.triu_indices(x.shape[0], 1)
    im = np.imag(a[iu[0]] * np.conj(a[iu[1]]))
    w = np.abs(im.mean(axis=-1)) / (np.abs(im).mean(axis=-1) + 1e-12)
    return float(np.median(w))


def msc_all_pairs(x, fs):
    """Mean magnitude-squared coherence over frequencies, median over channel pairs."""
    f, pxx = welch(x, fs=fs, axis=-1)
    iu = np.triu_indices(x.shape[0], 1)
    _, pxy = csd(x[iu[0]], x[iu[1]], fs=fs, axis=-1)
    coh = np.abs(pxy) ** 2 / (pxx[iu[0]] * pxx[iu[1]] + 1e-12)
    return float(np.median(coh.mean(axis=-1)))


def pdbsi(bp_band, pairs_idx):
    """Unsigned pairwise derived BSI per band, median over homologous pairs."""
    vals = [abs(bp_band[l] - bp_band[r]) / (bp_band[l] + bp_band[r] + 1e-12) for l, r in pairs_idx]
    return float(np.median(vals)) if vals else np.nan


def homologous_index_pairs(names):
    pos = {n.upper(): i for i, n in enumerate(names)}
    return [(pos[a], pos[b]) for a, b in C.HOMOLOGOUS_PAIRS if a in pos and b in pos]


# ----------------------------------------------------------------------------- one trial
def trial_features(x, fs, pairs_idx):
    """x: [ch, n] filtered trial -> dict of the 17 features."""
    bp = band_powers(x, fs)
    eps = 1e-12
    out = {b: float(np.median(bp[b])) for b in C.BANDS}
    out["delta_alpha"] = float(np.median(bp["delta"] / (bp["alpha"] + eps)))
    out["theta_alpha"] = float(np.median(bp["theta"] / (bp["alpha"] + eps)))
    out["dt_ab"] = float(np.median((bp["delta"] + bp["theta"]) / (bp["alpha"] + bp["beta"] + eps)))
    out["kurtosis"] = float(np.median(kurtosis(x, axis=-1, fisher=False)))
    out["skewness"] = float(np.median(skew(x, axis=-1, bias=False)))
    out["sample_entropy"] = float(np.nanmedian([sample_entropy(c) for c in x]))
    out["higuchi_fd"] = float(np.median([higuchi_fd(c) for c in x]))
    out["weighted_phase_lag_index"] = wpli_all_pairs(x)
    out["magnitude_squared_coherence"] = msc_all_pairs(x, fs)
    for b in C.BANDS:
        out[f"pdbsi_{b}"] = pdbsi(bp[b], pairs_idx)
    return {k: out[k] for k in C.FEATURES_17}
