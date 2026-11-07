"""Feature-selection strategies compared in Study A (classification) and Study C (regression).

Every strategy is fitted on training data only. Strategies with a tuning parameter choose it by an
inner cross-validation that is grouped by subject, so trials of one subject never inform their own
selection, using the one-standard-error rule (the sparsest setting within one SE of the best score). All strategies return a list of original feature names, except PCA, which returns
components and is included as a non-interpretable reference.

  all          all 17 features (no selection)
  panel        a-priori clinical panel (common.CLINICAL_PANEL)
  corr_vif     label-agnostic |r| screening + iterative VIF (pipeline/selection.py)
  cluster      label-agnostic hierarchical clustering on 1-|Spearman rho|, one representative per cluster
  mrmr         minimum-redundancy maximum-relevance (Peng et al. 2005), k tuned by inner CV
  enet         elastic-net regularised linear model; features with a non-zero coefficient
  glasso       group lasso over the six feature families (Yuan & Lin 2006); whole families in or out
  stability    stability selection (Meinshausen & Buhlmann 2010) with L1 models on half-subsamples
  rfe          recursive feature elimination with a random forest, size tuned by inner CV
  pca          principal components explaining 90% of the variance (reference, not a selection)
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

import common as K
import selection as S          # pipeline/selection.py (corr + VIF chain)

warnings.filterwarnings("ignore")
METHODS = ["all", "panel", "corr_vif", "cluster", "mrmr", "enet", "glasso", "stability", "rfe", "pca"]
METHOD_LABELS = {"all": "All 17", "panel": "Clinical panel", "corr_vif": "Correlation + VIF",
                 "cluster": "Correlation clustering", "mrmr": "mRMR", "enet": "Elastic net",
                 "glasso": "Group lasso", "stability": "Stability selection", "rfe": "RFE (random forest)",
                 "pca": "PCA (90% variance)"}
LABEL_AGNOSTIC = {"all", "panel", "corr_vif", "cluster", "pca"}
PRIORITY = S.PRIORITY


# ----------------------------------------------------------------------------- helpers
def _inner_splits(y, groups, task, n_splits=3, seed=K.SEED):
    n_groups = len(np.unique(groups))
    n_splits = min(n_splits, n_groups)
    if task == "clf":
        return list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups))
    return list(GroupKFold(n_splits).split(np.zeros(len(y)), y, groups))


def _quick_model(task):
    if task == "clf":
        return LogisticRegression(C=1.0, max_iter=3000)
    from sklearn.linear_model import Ridge
    return Ridge(alpha=1.0)


def _score(task, y, yhat):
    return f1_score(y, yhat, average="macro") if task == "clf" else -mean_absolute_error(y, yhat)


def _one_se(params, fold_scores, parsimony):
    """Most parsimonious parameter whose mean inner-CV score is within one standard error of the
    best (Hastie et al.'s one-standard-error rule). parsimony(p): smaller = sparser."""
    means = np.array([np.mean(s) for s in fold_scores])
    ses = np.array([np.std(s, ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0.0 for s in fold_scores])
    best = int(np.argmax(means))
    ok = [i for i in range(len(params)) if means[i] >= means[best] - ses[best]]
    return params[min(ok, key=lambda i: parsimony(params[i]))]


def _cv_scores_subset(X, y, groups, cols, task):
    sc = []
    for tr, va in _inner_splits(y, groups, task):
        s = StandardScaler().fit(X.iloc[tr][cols])
        m = _quick_model(task).fit(s.transform(X.iloc[tr][cols]), y[tr])
        sc.append(_score(task, y[va], m.predict(s.transform(X.iloc[va][cols]))))
    return sc


def _cv_score_subset(X, y, groups, cols, task):
    sc = []
    for tr, va in _inner_splits(y, groups, task):
        s = StandardScaler().fit(X.iloc[tr][cols])
        m = _quick_model(task).fit(s.transform(X.iloc[tr][cols]), y[tr])
        sc.append(_score(task, y[va], m.predict(s.transform(X.iloc[va][cols]))))
    return float(np.mean(sc))


def _z(X):
    s = StandardScaler().fit(X)
    return pd.DataFrame(s.transform(X), columns=X.columns, index=X.index)


# ----------------------------------------------------------------------------- label-agnostic
def select_cluster(X, thr=0.6):
    """Average-linkage clusters of 1-|rho|; cut where features correlate at |rho| >= thr."""
    r = X.corr(method="spearman").abs().fillna(0).to_numpy(copy=True)
    np.fill_diagonal(r, 1.0)
    dist = squareform(1 - r, checks=False)
    lab = fcluster(linkage(dist, "average"), t=1 - thr, criterion="distance")
    keep = []
    for c in np.unique(lab):
        members = [f for f, l in zip(X.columns, lab) if l == c]
        keep.append(min(members, key=lambda f: PRIORITY.index(f) if f in PRIORITY else 99))
    return [f for f in X.columns if f in keep]


# ----------------------------------------------------------------------------- mRMR
def mrmr_order(X, y, task, seed=K.SEED):
    """Greedy mRMR ranking with the mutual-information difference criterion (MID)."""
    Xz = _z(X)
    mi_fn = mutual_info_classif if task == "clf" else mutual_info_regression
    rel = pd.Series(mi_fn(Xz.values, y, random_state=seed), index=X.columns)
    cols = list(X.columns)
    red = pd.DataFrame(0.0, index=cols, columns=cols)
    for c in cols:   # feature-feature MI (kNN estimator)
        red[c] = mutual_info_regression(Xz.values, Xz[c].values, random_state=seed)
    order = [rel.idxmax()]
    while len(order) < len(cols):
        rest = [c for c in cols if c not in order]
        score = {c: rel[c] - red.loc[c, order].mean() for c in rest}
        order.append(max(score, key=score.get))
    return order


def select_mrmr(X, y, groups, task, ks=(3, 5, 7, 9, 11)):
    order = mrmr_order(X, y, task)
    ks = [k for k in ks if k <= len(order)]
    best = _one_se(ks, [_cv_scores_subset(X, y, groups, order[:k], task) for k in ks], parsimony=lambda k: k)
    return order[:best]


# ----------------------------------------------------------------------------- elastic net
def select_enet(X, y, groups, task):
    Xz = _z(X)
    grid = [(C, l1) for C in np.logspace(-2.5, 0.5, 6) for l1 in (0.5, 0.9)] if task == "clf" else \
           [(a, l1) for a in np.logspace(-2, 0.5, 6) for l1 in (0.5, 0.9)]

    def model(p):
        if task == "clf":
            return LogisticRegression(penalty="elasticnet", solver="saga", C=p[0], l1_ratio=p[1], max_iter=4000)
        return ElasticNet(alpha=p[0], l1_ratio=p[1], max_iter=20000)

    def scores(p):
        sc = []
        for tr, va in _inner_splits(y, groups, task):
            m = model(p).fit(Xz.iloc[tr], y[tr])
            sc.append(_score(task, y[va], m.predict(Xz.iloc[va])))
        return sc

    # sparser = stronger penalty: smaller C (classification) or larger alpha (regression)
    best = _one_se(grid, [scores(p) for p in grid],
                   parsimony=(lambda p: p[0]) if task == "clf" else (lambda p: -p[0]))
    coef = np.atleast_2d(model(best).fit(Xz, y).coef_)
    keep = [f for f, c in zip(X.columns, np.abs(coef).max(0)) if c > 1e-8]
    return keep or [X.columns[int(np.abs(coef).max(0).argmax())]]


# ----------------------------------------------------------------------------- group lasso
def _softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def group_lasso_fit(X, y, groups_idx, lam, task, max_iter=1000, tol=1e-6):
    """Proximal gradient (with backtracking) for multinomial-logistic or squared loss plus
    lam * sum_g sqrt(|g|) * ||W_g||_F. X standardized (n x p). Returns W (p x K) and intercept."""
    n, p = X.shape
    if task == "clf":
        classes = np.unique(y)
        Y = (y[:, None] == classes[None, :]).astype(float)
    else:
        Y = (y - y.mean())[:, None].astype(float)
    K_ = Y.shape[1]
    W, b = np.zeros((p, K_)), np.zeros(K_)
    if task != "clf":
        b = np.array([y.mean()])

    def loss_grad(W, b):
        Z = X @ W + b
        if task == "clf":
            P = _softmax(Z)
            loss = -np.mean(np.sum(Y * np.log(P + 1e-12), 1))
            G = (P - Y) / n
        else:
            R = Z - (Y + y.mean())
            loss = 0.5 * np.mean(R ** 2)
            G = R / n
        return loss, X.T @ G, G.sum(0)

    def penalty(W):
        return lam * sum(np.sqrt(len(g)) * np.linalg.norm(W[g]) for g in groups_idx)

    step = 1.0
    loss, gW, gb = loss_grad(W, b)
    for _ in range(max_iter):
        while True:
            Wn = W - step * gW
            for g in groups_idx:   # block soft-thresholding
                nrm = np.linalg.norm(Wn[g])
                Wn[g] = 0 if nrm <= step * lam * np.sqrt(len(g)) else Wn[g] * (1 - step * lam * np.sqrt(len(g)) / nrm)
            bn = b - step * gb
            ln, _, _ = loss_grad(Wn, bn)
            D = Wn - W
            if ln <= loss + np.sum(gW * D) + np.sum(gb * (bn - b)) + (np.sum(D ** 2) + np.sum((bn - b) ** 2)) / (2 * step):
                break
            step *= 0.5
        obj_old = loss + penalty(W)
        W, b = Wn, bn
        loss, gW, gb = loss_grad(W, b)
        if abs(obj_old - (loss + penalty(W))) < tol * max(1.0, abs(obj_old)):
            break
    return W, b


def select_glasso(X, y, groups, task, lams=None):
    Xz = _z(X)
    cols = list(X.columns)
    fam = [[cols.index(f) for f in fs if f in cols] for fs in K.FAMILIES.values()]
    fam = [g for g in fam if g]
    yv = np.asarray(y)
    if lams is None:   # grid from lambda_max (all groups zero) down to lambda_max / 100
        if task == "clf":
            Y = (yv[:, None] == np.unique(yv)[None, :]).astype(float)
            G = Xz.values.T @ (Y - Y.mean(0)) / len(yv)
        else:
            G = Xz.values.T @ (yv - yv.mean())[:, None] / len(yv)
        lam_max = max(np.linalg.norm(G[g]) / np.sqrt(len(g)) for g in fam)
        lams = list(np.geomspace(lam_max * 0.99, lam_max / 100, 8))

    def fit_predict(Xtr, ytr, Xte, lam):
        W, b = group_lasso_fit(Xtr, ytr, fam, lam, task)
        Z = Xte @ W + b
        if task == "clf":
            return np.unique(ytr)[Z.argmax(1)]
        return Z[:, 0]

    def scores(lam):
        return [_score(task, yv[va], fit_predict(Xz.values[tr], yv[tr], Xz.values[va], lam))
                for tr, va in _inner_splits(yv, groups, task)]

    best = _one_se(lams, [scores(lam) for lam in lams], parsimony=lambda lam: -lam)
    W, _ = group_lasso_fit(Xz.values, yv, fam, best, task)
    keep = [cols[i] for g in fam if np.linalg.norm(W[g]) > 1e-8 for i in g]
    return [c for c in cols if c in keep] or select_glasso(X, y, groups, task, lams=[min(lams) / 10])


# ----------------------------------------------------------------------------- stability selection
def select_stability(X, y, groups, task, n_sub=50, threshold=0.6, q=None, seed=K.SEED):
    """Stability selection (Meinshausen & Buhlmann 2010). For each of n_sub half-subsamples (drawn at
    the level of subjects) an L1 model is fitted along a regularisation grid. The grid is restricted to
    penalties that select on average at most q features (default q = round(sqrt(0.8 p))), which bounds
    the expected number of falsely selected features by q^2 / ((2*threshold - 1) p). A feature is kept
    if its maximum selection probability over the restricted grid reaches the threshold."""
    rng = np.random.default_rng(seed)
    Xz = _z(X)
    p = X.shape[1]
    q = q or max(1, int(round(np.sqrt(0.8 * p))))
    ug = np.unique(groups)
    grid = np.logspace(-3, 0.5, 10) if task == "clf" else np.logspace(0.5, -2.5, 10)
    sel = np.zeros((len(grid), n_sub, p), bool)
    for b in range(n_sub):
        sub = rng.choice(ug, size=max(2, len(ug) // 2), replace=False)
        m = np.isin(groups, sub)
        if task == "clf" and len(np.unique(y[m])) < 2:
            continue
        for j, c in enumerate(grid):
            if task == "clf":
                mdl = LogisticRegression(penalty="l1", solver="saga", C=c, max_iter=3000).fit(Xz[m], y[m])
            else:
                from sklearn.linear_model import Lasso
                mdl = Lasso(alpha=c, max_iter=20000).fit(Xz[m], y[m])
            sel[j, b] = np.abs(np.atleast_2d(mdl.coef_)).max(0) > 1e-8
    avg_k = sel.sum(2).mean(1)
    ok = avg_k <= q
    if not ok.any():
        ok[int(np.argmin(avg_k))] = True
    prob = sel[ok].mean(1).max(0)
    keep = [f for f, pr in zip(X.columns, prob) if pr >= threshold]
    return keep or list(X.columns[np.argsort(-prob)[:1]])


# ----------------------------------------------------------------------------- RFE
def select_rfe(X, y, groups, task, min_features=2, seed=K.SEED):
    est = RandomForestClassifier(200, random_state=seed, n_jobs=1) if task == "clf" else \
        RandomForestRegressor(200, random_state=seed, n_jobs=1)
    remaining = list(X.columns)
    path = []
    while len(remaining) >= min_features:
        path.append(list(remaining))
        if len(remaining) == min_features:
            break
        imp = pd.Series(est.fit(X[remaining], y).feature_importances_, index=remaining)
        remaining.remove(imp.idxmin())
    return _one_se(path, [_cv_scores_subset(X, y, groups, cols, task) for cols in path], parsimony=len)


# ----------------------------------------------------------------------------- dispatcher
def select(method, X, y, groups, task="clf"):
    """Return the selected feature list (or the string 'pca' for the PCA reference)."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    if method == "all":
        return list(X.columns)
    if method == "panel":
        return [f for f in K.CLINICAL_PANEL if f in X.columns]
    if method == "corr_vif":
        keep, _ = S.fit_redundancy(X)
        return keep
    if method == "cluster":
        return select_cluster(X)
    if method == "mrmr":
        return select_mrmr(X, y, groups, task)
    if method == "enet":
        return select_enet(X, y, groups, task)
    if method == "glasso":
        return select_glasso(X, y, groups, task)
    if method == "stability":
        return select_stability(X, y, groups, task)
    if method == "rfe":
        return select_rfe(X, y, groups, task)
    if method == "pca":
        return "pca"
    raise ValueError(method)


class PCAReference:
    """Standardize + PCA keeping 90% of the variance (fitted on training data)."""

    def __init__(self, var=0.90):
        self.var = var

    def fit(self, X):
        self.sc = StandardScaler().fit(X)
        self.pca = PCA(n_components=self.var, svd_solver="full").fit(self.sc.transform(X))
        return self

    def transform(self, X):
        return self.pca.transform(self.sc.transform(X))

    @property
    def n_components(self):
        return int(self.pca.n_components_)
