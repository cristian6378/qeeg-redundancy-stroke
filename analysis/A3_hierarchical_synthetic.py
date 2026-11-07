"""Study A, step 3 (review revision): harder, hierarchical, graded-collinearity benchmark.

The thesis benchmark assigns single-class "subjects" to blocks of stored trials, is almost perfectly separable
and has a maximum VIF of 2.9. This generator is feature-level but explicitly hierarchical:

  x_ij = mu_class * delta * s_i * w  +  b_i  +  e_ij
    class    : normal, TIA, ischaemic, haemorrhagic with ordered mean shifts (0, 0.3, 1.0, 1.2) along a direction w
               that loads on slowing (delta, theta, ratios, pdBSI up; alpha, beta down) and weakly on the rest
    delta    : global effect size (difficulty); s_i ~ N(1, 0.5): subject-specific lesion severity
    b_i      : subject random effect (SD tau) shared by all trials of the subject
    e_ij     : trial noise with a graded correlation structure: within-family correlation rho, cross-family 0.3*rho
    ratios   : log ratios are noisy linear functions of the log band powers (residual SD sqrt(1-rho))

so that subjects are independent units, classes overlap at low delta, and collinearity is varied (rho).
The strategies of fs_methods are run in subject-grouped 5-fold CV with the logistic classifier (LightGBM is not
needed). Reported per scenario and strategy: macro-F1, number of inputs, Nogueira stability, and the maximum
VIF of the generated data.

  python A3_hierarchical_synthetic.py [--deltas 0.3 0.6 1.0] [--rhos 0.3 0.6 0.9] [--subjects 120]
                                      [--trials 10] [--repeats 1] [--n-jobs -1] [--methods ...] [--fast]
Outputs (MDPI_RESULTS/studyA3/): results.csv, selections.jsonl, scenario_info.csv
"""
import argparse
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

import common as K
import fs_methods as SEL
import measures as M

OUT = K.OUT / "studyA3"
CLASS_SHIFT = {"normal": 0.0, "tia": 0.3, "ischaemic": 1.0, "haemorrhagic": 1.2}
UP = ["delta", "theta", "delta_alpha", "theta_alpha", "dt_ab", "pdbsi_delta", "pdbsi_theta", "pdbsi_alpha", "pdbsi_beta"]
DOWN = ["alpha", "beta"]


def direction():
    w = np.zeros(len(K.F17))
    for i, f in enumerate(K.F17):
        w[i] = 1.0 if f in UP else (-1.0 if f in DOWN else 0.15)
    return w / np.linalg.norm(w) * np.sqrt(len(UP) + len(DOWN))


def noise_cov(rho):
    n = len(K.F17)
    C = np.eye(n)
    fam = [K.FAMILY_OF[f] for f in K.F17]
    for i in range(n):
        for j in range(n):
            if i != j:
                C[i, j] = rho if fam[i] == fam[j] else 0.3 * rho
    # ratios are built from band powers: add dependence between spectral and ratio families
    for i, fi in enumerate(K.F17):
        for j, fj in enumerate(K.F17):
            if {K.FAMILY_OF[fi], K.FAMILY_OF[fj]} == {"spectral", "ratio"}:
                C[i, j] = max(C[i, j], 0.5 * rho)
    ev, evec = np.linalg.eigh(C)
    ev = np.clip(ev, 1e-3, None)
    return evec @ np.diag(ev) @ evec.T


IDX = {f: i for i, f in enumerate(K.F17)}


def derive_ratios(x, rho, rng):
    """Log ratios are (noisy) linear functions of the log band powers, as in real qEEG; the residual noise
    shrinks with rho, so collinearity is graded from moderate (rho 0.3) to near-exact (rho 0.9)."""
    x = x.copy()
    eps = np.sqrt(max(1 - rho, 1e-3))
    d, th, a, be = (x[IDX[f]] for f in ("delta", "theta", "alpha", "beta"))
    x[IDX["delta_alpha"]] = (d - a) / np.sqrt(2) + eps * rng.standard_normal()
    x[IDX["theta_alpha"]] = (th - a) / np.sqrt(2) + eps * rng.standard_normal()
    x[IDX["dt_ab"]] = 0.5 * (d + th - a - be) + eps * rng.standard_normal()
    return x


def generate(delta, rho, n_subjects=120, n_trials=10, tau=0.6, sigma=1.0, seed=K.SEED):
    rng = np.random.default_rng(seed)
    classes = list(CLASS_SHIFT)
    per = n_subjects // len(classes)
    w = direction()
    L = np.linalg.cholesky(noise_cov(rho))
    rows = []
    sid = 0
    for c in classes:
        for _ in range(per):
            sid += 1
            s = rng.normal(1.0, 0.5)
            b = tau * (L @ rng.standard_normal(len(K.F17)))
            for t in range(n_trials):
                e = (L @ rng.standard_normal(len(K.F17))) * sigma
                x = CLASS_SHIFT[c] * delta * s * w + b + e
                x = derive_ratios(x, rho, rng)
                rows.append({"subject_id": sid, "trial_id": t, "label": c, **dict(zip(K.F17, x))})
    return pd.DataFrame(rows)


def max_vif(df):
    X = df[K.F17].values
    X = (X - X.mean(0)) / X.std(0)
    R = np.linalg.inv(np.corrcoef(X.T))
    return float(np.max(np.diag(R)))


def run_scenario(delta, rho, args):
    import A1_selection_synthetic as A1
    A1.CLASSIFIERS = {"Logistic Regression": A1.fit_lr}
    df = generate(delta, rho, args.subjects, args.trials)
    classes = sorted(df.label.unique())
    y = df.label.map({c: i for i, c in enumerate(classes)}).values
    g = df.subject_id.values
    jobs = []
    for rep in range(args.repeats):
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=K.SEED + rep)
        for k, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y, g)):
            jobs.append((rep, k, tr, te))
    res = Parallel(n_jobs=args.n_jobs)(delayed(A1.run_fold)(df, y, g, rep, k, tr, te, args.methods) for rep, k, tr, te in jobs)
    preds = pd.concat([r[0] for r in res])
    sels = [s for r in res for s in r[1]]
    rows = []
    for m in args.methods:
        p = preds[preds.method == m]
        f1 = [f1_score(d.y_true, d.y_pred, average="macro") for _, d in p.groupby(["repeat", "fold"])]
        S = [s for s in sels if s["method"] == m]
        Z = np.array([[f in s["selected"] for f in K.F17] for s in S], float)
        rows.append({"delta": delta, "rho": rho, "method": m, "macro_f1": float(np.mean(f1)), "macro_f1_sd": float(np.std(f1)),
                     "n_inputs": float(np.mean([s["n_inputs"] for s in S])),
                     "stability": M.nogueira_stability(Z)["phi"] if len(Z) > 1 else np.nan})
    return rows, sels, {"delta": delta, "rho": rho, "max_vif": max_vif(df), "n_rows": len(df)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.3, 0.6, 1.0])
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.3, 0.6, 0.9])
    ap.add_argument("--subjects", type=int, default=120)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--methods", nargs="+", default=SEL.METHODS)
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    if args.fast:
        args.deltas, args.rhos, args.subjects, args.trials = [0.6], [0.6], 40, 6
    K.ensure_dirs(); OUT.mkdir(parents=True, exist_ok=True)
    allrows, allsels, info = [], [], []
    for d in args.deltas:
        for r in args.rhos:
            rows, sels, inf = run_scenario(d, r, args)
            allrows += rows; allsels += [{**s, "delta": d, "rho": r} for s in sels]; info.append(inf)
            print(f"delta {d} rho {r}: max VIF {inf['max_vif']:.1f}", flush=True)
    pd.DataFrame(allrows).to_csv(OUT / "results.csv", index=False)
    pd.DataFrame(info).to_csv(OUT / "scenario_info.csv", index=False)
    with open(OUT / "selections.jsonl", "w") as f:
        for s in allsels:
            f.write(json.dumps(s) + "\n")
    print("done ->", OUT)


if __name__ == "__main__":
    main()
