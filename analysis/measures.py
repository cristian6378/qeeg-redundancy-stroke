"""Reliability, agreement and stability measures used in Studies A-C."""
import itertools

import numpy as np
from scipy import stats


def icc_oneway(Y, alpha=0.05):
    """One-way random-effects ICC (Shrout & Fleiss 1979, case 1) for an n x k matrix
    (targets x repeated measurements). Returns ICC(1,1), ICC(1,k) and F-based CIs for ICC(1,1)."""
    Y = np.asarray(Y, float)
    Y = Y[~np.isnan(Y).any(axis=1)]
    n, k = Y.shape
    grand = Y.mean()
    msb = k * ((Y.mean(1) - grand) ** 2).sum() / (n - 1)
    msw = ((Y - Y.mean(1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    icc1 = (msb - msw) / (msb + (k - 1) * msw)
    icck = (msb - msw) / msb
    f0 = msb / msw
    fl = f0 / stats.f.ppf(1 - alpha / 2, n - 1, n * (k - 1))
    fu = f0 * stats.f.ppf(1 - alpha / 2, n * (k - 1), n - 1)
    return dict(icc1=float(icc1), icck=float(icck), lo=float((fl - 1) / (fl + k - 1)),
                hi=float((fu - 1) / (fu + k - 1)), n=n, k=k, F=float(f0))


def icc_twoway(Y):
    """ICC(2,1) and ICC(3,1) (Shrout & Fleiss cases 2 and 3) for an n x k matrix."""
    Y = np.asarray(Y, float)
    n, k = Y.shape
    grand = Y.mean()
    ssr = k * ((Y.mean(1) - grand) ** 2).sum()
    ssc = n * ((Y.mean(0) - grand) ** 2).sum()
    sse = ((Y - grand) ** 2).sum() - ssr - ssc
    msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    return dict(icc2=float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)),
                icc3=float((msr - mse) / (msr + (k - 1) * mse)))


def spearman_brown_k(icc1, target):
    """Number of averaged trials needed for the mean to reach a target reliability."""
    if icc1 <= 0:
        return np.inf
    return float(target * (1 - icc1) / (icc1 * (1 - target)))


def ccc(x, y):
    """Lin's concordance correlation coefficient."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    sxy = np.mean((x - x.mean()) * (y - y.mean()))
    return float(2 * sxy / (x.var() + y.var() + (x.mean() - y.mean()) ** 2))


def bootstrap_ci(fn, *arrays, n_boot=1000, seed=0, alpha=0.05):
    """Percentile CI of fn over resampled rows (patients)."""
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        try:
            vals.append(fn(*[np.asarray(a)[i] for a in arrays]))
        except (ValueError, ZeroDivisionError, FloatingPointError):
            continue
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    return (float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))) \
        if len(vals) else (np.nan, np.nan)


def nogueira_stability(Z):
    """Stability of feature selection (Nogueira, Sechidis & Brown 2018). Z: M x d binary matrix
    (M selection runs, d features). Returns the estimate and its asymptotic 95% CI."""
    Z = np.asarray(Z, float)
    M, d = Z.shape
    p = Z.mean(0)
    kbar = Z.sum(1).mean()
    denom = (kbar / d) * (1 - kbar / d)
    if denom == 0:
        return dict(phi=1.0, lo=1.0, hi=1.0)
    s2 = M / (M - 1) * p * (1 - p)
    phi = 1 - s2.mean() / denom
    # variance estimator of Nogueira et al. (2018), as in their reference implementation
    k = Z.sum(1)
    phis = (1 / denom) * ((Z * p).mean(1) - k * kbar / d ** 2
                          + (phi / 2) * (2 * k * kbar / d ** 2 - k / d - kbar / d + 1))
    se = np.sqrt(4 / M ** 2 * ((phis - phis.mean()) ** 2).sum())
    return dict(phi=float(phi), lo=float(phi - 1.96 * se), hi=float(phi + 1.96 * se))


def mean_jaccard(sets):
    sets = [set(s) for s in sets]
    vals = [len(a & b) / len(a | b) if (a | b) else 1.0 for a, b in itertools.combinations(sets, 2)]
    return float(np.mean(vals)) if vals else np.nan


def partial_spearman(x, y, covars):
    """Spearman correlation of x and y after removing the rank-linear effect of covariates."""
    x, y = stats.rankdata(x), stats.rankdata(y)
    if covars is None or len(covars) == 0:
        return stats.pearsonr(x, y)
    Z = np.column_stack([np.ones(len(x))] + [stats.rankdata(c) for c in covars])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    r = np.corrcoef(rx, ry)[0, 1]
    dof = len(x) - 2 - (Z.shape[1] - 1)
    t = r * np.sqrt(dof / max(1e-12, 1 - r ** 2))
    return r, float(2 * stats.t.sf(abs(t), dof))
