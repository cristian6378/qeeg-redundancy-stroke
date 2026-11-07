"""Study A, step 1: compare feature-selection strategies on the synthetic four-class benchmark.

The synthetic feature table (electra_features_realistic.csv, thesis repository) is perturbed exactly as
in the thesis analysis notebook (class overlap, grouped correlated noise, outliers; seed 42) to obtain
the 'realistic' condition. Ten strategies (fs_methods.METHODS) are each fitted inside every training
fold of a subject-grouped 5-fold cross-validation repeated 5 times (25 outer folds), followed by two
downstream classifiers: L2 logistic regression (C tuned by inner grouped CV) and gradient-boosted
trees (LightGBM, fixed settings).

  python A1_selection_synthetic.py [--condition realistic] [--repeats 5] [--n-jobs -1] [--fast]
Outputs (MDPI_RESULTS/studyA/): predictions.csv, selections.jsonl, run_info.json
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

import common as K
import fs_methods as SEL

SCENARIOS = {"clean": dict(overlap=0.0, outlier=0.0, noise=0.0),
             "mild": dict(overlap=0.15, outlier=0.03, noise=0.03),
             "realistic": dict(overlap=0.30, outlier=0.06, noise=0.06),
             "severe": dict(overlap=0.45, outlier=0.1, noise=0.1)}
NOISE_GROUPS = [["delta", "theta", "alpha", "beta"], ["delta_alpha", "theta_alpha", "dt_ab"],
                ["pdbsi_delta", "pdbsi_theta", "pdbsi_alpha", "pdbsi_beta"]]


# ----------------------------------------------------------------------------- data (thesis protocol)
def _overlap(df, strength, feats, rng):
    out = df.copy()
    for label in sorted(df["label"].unique()):
        mask = df["label"] == label
        other = df[df["label"] != label]
        for f in feats:
            mixed = other[f].sample(mask.sum(), replace=True, random_state=rng.integers(0, 1e9)).values
            out.loc[mask, f] = (1 - strength) * df.loc[mask, f].values + strength * mixed
    return out


def _grouped_noise(df, noise, feats, random_state=42):
    rng = np.random.default_rng(random_state)
    out = df.copy()
    for group in NOISE_GROUPS:
        shared = rng.normal(0, 1, size=len(out))
        for f in group:
            if f not in feats:
                continue
            s = out[f].std(skipna=True)
            ind = rng.normal(0, 1, size=len(out))
            out[f] = out[f] + noise * s * (0.8 * shared + 0.2 * ind)
    return out


def _outliers(df, ratio, feats, rng):
    out = df.copy()
    idxs = rng.choice(df.index, int(len(df) * ratio), replace=False)
    for i in idxs:
        for f in rng.choice(feats, size=3, replace=False):
            out.loc[i, f] += rng.choice([-1, 1]) * 4 * df[f].std()
    return out


def load_synthetic(condition="realistic", path=K.SYNTH_CSV):
    """Reproduces the thesis notebook (analyses.ipynb, cells 2-14): subject ids are blocks of 50 trials,
    scenarios are generated in the order clean, mild, realistic, severe from one generator (seed 42)."""
    if not path.exists():
        raise SystemExit(f"synthetic feature table not found: {path}\nSet SYNTH_FEATURES or copy "
                         "electra_features_realistic.csv from the thesis repository to data/synthetic/")
    df = pd.read_csv(path).rename(columns={"higuchi_fractal_dimension": "higuchi_fd"})
    missing = [f for f in K.F17 if f not in df.columns]
    if missing:
        raise SystemExit(f"synthetic table lacks features {missing}")
    df.insert(0, "trial_id", np.tile(np.arange(1, 51), len(df) // 50))
    df.insert(0, "subject_id", np.repeat(np.arange(1, len(df) // 50 + 1), 50))
    feats = [c for c in df.columns if c not in ("subject_id", "trial_id", "label")]
    rng = np.random.default_rng(42)
    out = None
    for name, p in SCENARIOS.items():
        d = _overlap(df.copy(), p["overlap"], feats, rng)
        d = _grouped_noise(d, p["noise"], feats)
        d = _outliers(d, p["outlier"], feats, rng)
        if name == condition:
            out = d
    out = out.dropna().reset_index(drop=True)
    return out


# ----------------------------------------------------------------------------- models
def make_lgbm():
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, random_state=K.SEED, n_jobs=1, verbose=-1)
    except ImportError:   # smoke tests only
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=200, random_state=K.SEED)


def fit_lr(Xtr, ytr, gtr):
    """L2 logistic regression; C by inner grouped 3-fold CV (macro-F1)."""
    best, best_s = 1.0, -1
    for C in (0.01, 0.1, 1.0, 10.0, 100.0):
        sc = []
        for tr, va in SEL._inner_splits(ytr, gtr, "clf"):
            s = StandardScaler().fit(Xtr[tr])
            m = LogisticRegression(C=C, max_iter=5000).fit(s.transform(Xtr[tr]), ytr[tr])
            sc.append(f1_score(ytr[va], m.predict(s.transform(Xtr[va])), average="macro"))
        if np.mean(sc) > best_s:
            best, best_s = C, np.mean(sc)
    s = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=best, max_iter=5000).fit(s.transform(Xtr), ytr)
    return lambda X: m.predict(s.transform(X)), {"C": best}


def fit_gbt(Xtr, ytr, gtr):
    m = make_lgbm().fit(Xtr, ytr)
    return m.predict, {}


CLASSIFIERS = {"Logistic Regression": fit_lr, "LightGBM": fit_gbt}


# ----------------------------------------------------------------------------- one outer fold
def run_fold(df, y, g, rep, k, tr, te, methods):
    X = df[K.F17]
    preds, sels = [], []
    for method in methods:
        t0 = time.perf_counter()
        sel = SEL.select(method, X.iloc[tr].reset_index(drop=True), y[tr], g[tr], "clf")
        t_sel = time.perf_counter() - t0
        if sel == "pca":
            pca = SEL.PCAReference().fit(X.iloc[tr].values)
            Xtr, Xte, chosen = pca.transform(X.iloc[tr].values), pca.transform(X.iloc[te].values), list(K.F17)
            n_feat = pca.n_components
        else:
            Xtr, Xte, chosen, n_feat = X.iloc[tr][sel].values, X.iloc[te][sel].values, sel, len(sel)
        sels.append({"repeat": rep, "fold": k, "method": method, "selected": chosen, "n_inputs": n_feat,
                     "selection_s": t_sel})
        for cname, fit in CLASSIFIERS.items():
            predict, hp = fit(Xtr, y[tr], g[tr])
            yhat = predict(Xte)
            preds.append(pd.DataFrame({"repeat": rep, "fold": k, "method": method, "classifier": cname,
                                       "subject": g[te], "y_true": y[te], "y_pred": yhat}))
    return pd.concat(preds), sels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="realistic", choices=list(SCENARIOS))
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--methods", nargs="+", default=SEL.METHODS)
    ap.add_argument("--fast", action="store_true", help="smoke test: 1 repeat")
    args = ap.parse_args()
    K.ensure_dirs()
    out = K.OUT / "studyA"
    out.mkdir(exist_ok=True)

    df = load_synthetic(args.condition)
    classes = sorted(df["label"].unique())
    y = df["label"].map({c: i for i, c in enumerate(classes)}).values
    g = df["subject_id"].values
    print(f"synthetic '{args.condition}': {len(df)} trials, {len(np.unique(g))} subjects, classes {classes}")
    reps = 1 if args.fast else args.repeats
    jobs = []
    for rep in range(reps):
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=K.SEED + rep)
        for k, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y, g)):
            assert not set(g[tr]) & set(g[te])
            jobs.append((rep, k, tr, te))
    res = Parallel(n_jobs=args.n_jobs, verbose=5)(delayed(run_fold)(df, y, g, rep, k, tr, te, args.methods)
                                                 for rep, k, tr, te in jobs)
    pd.concat([r[0] for r in res]).to_csv(out / "predictions.csv", index=False)
    with open(out / "selections.jsonl", "w") as f:
        for r in res:
            for s in r[1]:
                f.write(json.dumps(s) + "\n")
    df[["subject_id", "trial_id", "label"] + K.F17].to_csv(out / "synthetic_used.csv", index=False)
    (out / "run_info.json").write_text(json.dumps({"condition": args.condition, "repeats": reps, "folds": 5,
                                                   "n_trials": len(df), "classes": classes,
                                                   "methods": args.methods}, indent=2))
    print("done ->", out)


if __name__ == "__main__":
    main()
