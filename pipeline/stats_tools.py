"""Statistics used by 11_stats_extended.py (NumPy/SciPy only, so no extra dependency).

Repeated-measures designs: the unit is the patient; every patient is measured under every
condition (method, model x feature set, ...), so the data are a complete subjects x conditions
table. Implemented:
  rm_anova_1way / rm_anova_2way  univariate RM-ANOVA, Mauchly's sphericity test,
                                 Greenhouse-Geisser (GG) and Huynh-Feldt (HF) corrections,
                                 partial and generalized eta squared
  friedman                        Friedman chi^2, Iman-Davenport F, Kendall's W
  nemenyi                         all-pairs Nemenyi test and critical difference (Demsar, 2006)
  pairwise                        paired t-test and Wilcoxon signed-rank for all pairs,
                                  Holm or Benjamini-Hochberg correction, Cohen's d_z, 95% CI
  cd_diagram                      critical-difference diagram
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats


# ============================================================================ helpers
def _orth_contrasts(k):
    """(k-1) x k orthonormal contrasts (rows orthogonal to the unit vector)."""
    h = np.eye(k) - 1.0 / k
    u, s, _ = np.linalg.svd(h)
    return u[:, : k - 1].T


def sphericity(Y, C):
    """Mauchly's W, chi2, p, and GG / HF epsilons for contrasts C applied to Y (n x k)."""
    n = Y.shape[0]
    Z = Y @ C.T
    S = np.cov(Z, rowvar=False, ddof=1)
    S = np.atleast_2d(S)
    p = S.shape[0]
    tr = np.trace(S)
    eps_gg = tr ** 2 / (p * np.trace(S @ S)) if p > 0 else 1.0
    eps_hf = min(1.0, (n * p * eps_gg - 2) / (p * (n - 1 - p * eps_gg))) if p > 0 else 1.0
    if p < 2:
        return dict(W=1.0, chi2=0.0, p_mauchly=1.0, eps_gg=1.0, eps_hf=1.0)
    W = np.linalg.det(S) / (tr / p) ** p
    d = 1 - (2 * p ** 2 + p + 2) / (6 * p * (n - 1))
    chi2 = -(n - 1) * d * np.log(max(W, 1e-300))
    df = p * (p + 1) / 2 - 1
    return dict(W=float(W), chi2=float(chi2), p_mauchly=float(stats.chi2.sf(chi2, df)),
                eps_gg=float(eps_gg), eps_hf=float(max(eps_hf, eps_gg)))


def _effect_row(name, ss, df1, ss_err, df2, sph, ss_gen_denominator):
    F = (ss / df1) / (ss_err / df2)
    out = dict(effect=name, SS=ss, df1=df1, df2=df2, F=F, p=stats.f.sf(F, df1, df2),
               eta2_partial=ss / (ss + ss_err), eta2_generalized=ss / ss_gen_denominator)
    out.update(sph)
    out["p_GG"] = stats.f.sf(F, df1 * sph["eps_gg"], df2 * sph["eps_gg"])
    out["p_HF"] = stats.f.sf(F, df1 * sph["eps_hf"], df2 * sph["eps_hf"])
    return out


# ============================================================================ RM-ANOVA
def rm_anova_1way(Y):
    """Y: n subjects x k conditions (complete). Returns a one-row DataFrame."""
    Y = np.asarray(Y, float)
    n, k = Y.shape
    M = Y.mean()
    ss_cond = n * ((Y.mean(0) - M) ** 2).sum()
    ss_subj = k * ((Y.mean(1) - M) ** 2).sum()
    ss_tot = ((Y - M) ** 2).sum()
    ss_err = ss_tot - ss_cond - ss_subj
    sph = sphericity(Y, _orth_contrasts(k))
    row = _effect_row("condition", ss_cond, k - 1, ss_err, (k - 1) * (n - 1), sph,
                      ss_cond + ss_subj + ss_err)
    return pd.DataFrame([row])


def rm_anova_2way(Y, names=("A", "B")):
    """Y: n x a x b array (subjects x factor A x factor B), complete and balanced."""
    Y = np.asarray(Y, float)
    n, a, b = Y.shape
    M = Y.mean()
    ma, mb = Y.mean((0, 2)), Y.mean((0, 1))
    ms = Y.mean((1, 2))
    mab = Y.mean(0)
    mas, mbs = Y.mean(2), Y.mean(1)                    # n x a, n x b
    ss_a = n * b * ((ma - M) ** 2).sum()
    ss_b = n * a * ((mb - M) ** 2).sum()
    ss_ab = n * ((mab - ma[:, None] - mb[None, :] + M) ** 2).sum()
    ss_s = a * b * ((ms - M) ** 2).sum()
    ss_as = b * ((mas - ma[None, :] - ms[:, None] + M) ** 2).sum()
    ss_bs = a * ((mbs - mb[None, :] - ms[:, None] + M) ** 2).sum()
    resid = (Y - mab[None] - mas[:, :, None] - mbs[:, None, :]
             + ma[None, :, None] + mb[None, None, :] + ms[:, None, None] - M)
    ss_abs = (resid ** 2).sum()
    gen = ss_s + ss_as + ss_bs + ss_abs              # generalized eta^2 (all factors manipulated)
    Ca, Cb = _orth_contrasts(a), _orth_contrasts(b)
    rows = [
        _effect_row(names[0], ss_a, a - 1, ss_as, (a - 1) * (n - 1), sphericity(Y.mean(2), Ca), gen + ss_a),
        _effect_row(names[1], ss_b, b - 1, ss_bs, (b - 1) * (n - 1), sphericity(Y.mean(1), Cb), gen + ss_b),
        _effect_row(f"{names[0]} x {names[1]}", ss_ab, (a - 1) * (b - 1), ss_abs, (a - 1) * (b - 1) * (n - 1),
                    sphericity(Y.reshape(n, a * b), np.kron(Ca, Cb)), gen + ss_ab),
    ]
    return pd.DataFrame(rows)


def shapiro_residuals_1way(Y):
    Y = np.asarray(Y, float)
    r = Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + Y.mean()
    r = r.ravel()
    return stats.shapiro(r[:5000])


# ============================================================================ nonparametric
def friedman(Y):
    Y = np.asarray(Y, float)
    n, k = Y.shape
    chi2, p = stats.friedmanchisquare(*Y.T)
    W = chi2 / (n * (k - 1))
    ff = (n - 1) * chi2 / (n * (k - 1) - chi2) if n * (k - 1) - chi2 > 0 else np.inf
    p_id = stats.f.sf(ff, k - 1, (k - 1) * (n - 1))
    return dict(n=n, k=k, chi2=float(chi2), p=float(p), kendall_W=float(W),
                iman_davenport_F=float(ff), p_iman_davenport=float(p_id))


def mean_ranks(Y, higher_is_better=True):
    Y = np.asarray(Y, float)
    R = stats.rankdata(-Y if higher_is_better else Y, axis=1)   # rank 1 = best
    return R.mean(0)


def nemenyi(Y, labels, alpha=0.05, higher_is_better=True):
    Y = np.asarray(Y, float)
    n, k = Y.shape
    r = mean_ranks(Y, higher_is_better)
    se = np.sqrt(k * (k + 1) / (6.0 * n))
    q_alpha = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = q_alpha * se
    rows = []
    for i, j in itertools.combinations(range(k), 2):
        z = abs(r[i] - r[j]) / se
        rows.append(dict(a=labels[i], b=labels[j], rank_a=r[i], rank_b=r[j],
                         rank_diff=r[i] - r[j], p_nemenyi=float(stats.studentized_range.sf(z * np.sqrt(2), k, np.inf)),
                         significant=abs(r[i] - r[j]) > cd))
    return pd.DataFrame(rows), dict(zip(labels, r)), float(cd)


# ============================================================================ pairwise
def p_adjust(p, method="holm"):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    if method == "holm":
        run = 0.0
        for rank, i in enumerate(order):
            run = max(run, min(1.0, (m - rank) * p[i]))
            adj[i] = run
    elif method in ("fdr_bh", "bh"):
        run = 1.0
        for rank in range(m - 1, -1, -1):
            i = order[rank]
            run = min(run, p[i] * m / (rank + 1))
            adj[i] = min(run, 1.0)
    elif method == "bonferroni":
        adj = np.minimum(1.0, p * m)
    else:
        raise ValueError(method)
    return adj


def pairwise(Y, labels, method="holm", pairs=None):
    """All (or given) pairs: paired t-test, Wilcoxon, Cohen's d_z, mean difference with 95% CI."""
    Y = np.asarray(Y, float)
    n = Y.shape[0]
    idx = {l: i for i, l in enumerate(labels)}
    pairs = pairs or list(itertools.combinations(labels, 2))
    rows = []
    for a, b in pairs:
        d = Y[:, idx[a]] - Y[:, idx[b]]
        sd = d.std(ddof=1)
        t = d.mean() / (sd / np.sqrt(n)) if sd > 0 else 0.0
        p_t = 2 * stats.t.sf(abs(t), n - 1) if sd > 0 else 1.0
        p_w = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
        rows.append(dict(a=a, b=b, mean_diff=d.mean(), ci_lo=d.mean() - h, ci_hi=d.mean() + h,
                         t=t, p_ttest=p_t, p_wilcoxon=p_w, cohen_dz=d.mean() / sd if sd > 0 else 0.0))
    df = pd.DataFrame(rows)
    if len(df):
        df[f"p_ttest_{method}"] = p_adjust(df.p_ttest, method)
        df[f"p_wilcoxon_{method}"] = p_adjust(df.p_wilcoxon, method)
    return df


# ============================================================================ figure
def cd_diagram(ranks: dict, cd: float, title, path, nemenyi_df=None):
    """Demsar-style critical-difference diagram (lower rank = better)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = sorted(ranks.items(), key=lambda kv: kv[1])
    k = len(items)
    fig, ax = plt.subplots(figsize=(8, 0.35 * k + 1.6))
    ax.set_xlim(0.5, k + 0.5); ax.set_ylim(0.5, k + 2); ax.invert_xaxis()
    ax.hlines(k + 1, 1, k, color="black", lw=1)
    for r in range(1, k + 1):
        ax.vlines(r, k + 0.9, k + 1.1, color="black", lw=1)
        ax.text(r, k + 1.3, str(r), ha="center", fontsize=8)
    for i, (lab, r) in enumerate(items):
        y = k - 0.6 - 0.85 * i * (k - 0.6) / max(k - 1, 1) / 1.0 if k > 1 else k - 0.6
        y = max(y, 0.7)
        ax.plot([r, r], [k + 1, y], color="#1f4e79", lw=0.8)
        ax.plot([r, k + 0.45 if i < k / 2 else 0.55], [y, y], color="#1f4e79", lw=0.8)
        ax.text(k + 0.5 if i < k / 2 else 0.5, y, f"{lab} ({r:.2f})", fontsize=7,
                ha="right" if i < k / 2 else "left", va="center")
    ax.plot([1, 1 + cd], [k + 1.7, k + 1.7], color="red", lw=2)
    ax.text(1 + cd / 2, k + 1.85, f"CD = {cd:.2f}", ha="center", color="red", fontsize=8)
    # bars joining groups whose rank difference is within CD (maximal cliques only)
    cliques = []
    for i in range(k):
        j = i
        while j + 1 < k and items[j + 1][1] - items[i][1] <= cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    for c, (i, j) in enumerate(cliques):
        yb = k + 0.75 - 0.22 * c
        ax.plot([items[i][1] - 0.03, items[j][1] + 0.03], [yb, yb], color="black", lw=2.5)
    ax.axis("off"); ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)
