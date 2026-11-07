"""Study C: do the qEEG features carry information about clinical severity, and which selection
strategy recovers it?

Patient-level analysis (n = 47; one value per patient = mean of the 40 four-second, 29-channel trials
of features_variants.csv). Clinical variables are read from participants.tsv of the dataset; the
column names are detected automatically and printed (check them once).

  C-a association : Spearman and partial Spearman (age, and days since stroke when available)
                    correlations of each feature with each severity score, Benjamini-Hochberg FDR
  C-b prediction  : the primary score (NIHSS if present, otherwise Barthel/MBI, otherwise mRS) predicted by
                    ridge regression and a random forest after each selection strategy; repeated 5-fold CV
                    (20 repeats); Spearman rho and MAE out-of-fold; permutation test of the whole pipeline
  C-c montage     : the strongest associations recomputed with the sparse montages of Study B

  python C1_clinical_severity.py --inspect          (check the detected clinical columns first)
  python C1_clinical_severity.py [--repeats 20] [--n-perm 100] [--n-jobs -1]
Outputs (MDPI_RESULTS/studyC/): clinical_columns.json, association.csv, prediction.csv,
prediction_selection.csv, montage_association.csv, table_C*.tex, fig_association.png
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import common as K
import config as C
import fs_methods as SEL
import measures as M
import stats_tools as T

OUT = K.OUT / "studyC"
PATTERNS = {"nihss": r"nihss", "mbi": r"\bmbi\b|barthel|^mbi", "mrs": r"\bmrs\b|rankin|^mrs",
            "days": r"day|onset|course|duration|time.?after|since", "age": r"^age$|\bage\b", "sex": r"gender|sex"}
SEVERITY = ["nihss", "mbi", "mrs"]


# ----------------------------------------------------------------------------- clinical table
def clinical_table():
    cands = list(C.DATA_ROOT.rglob("participants.tsv")) + list(C.DATA_ROOT.rglob("participants.csv"))
    if not cands:
        raise SystemExit("participants.tsv not found under the data folder")
    p = cands[0]
    df = pd.read_csv(p, sep="\t" if p.suffix == ".tsv" else None, engine="python")
    cols = {}
    for key, pat in PATTERNS.items():
        hit = [c for c in df.columns if re.search(pat, c.strip().lower().replace("_", " "))]
        if hit:
            cols[key] = hit[0]
    sub_col = next(c for c in df.columns if re.search(r"participant|subject|^id$|sub", c.lower()))
    out = pd.DataFrame({"subject": df[sub_col].astype(str).str.extract(r"(\d+)")[0].astype(int)})
    for key, c in cols.items():
        v = df[c]
        out[key] = pd.to_numeric(v, errors="coerce") if key != "sex" else v.astype(str).str.lower().str[0].map(
            {"m": 0, "f": 1, "男": 0, "女": 1})
    print(f"[clinical] {p.name}: columns {list(df.columns)}\n[clinical] detected: {cols}")
    return out.set_index("subject"), cols, str(p)


# ----------------------------------------------------------------------------- features
def patient_features(montage="M29"):
    V = pd.read_csv(K.WORK / "features_variants.csv")
    d = V[(V.kind == "length") & (V.length_s == 4)] if montage == "M29" else V[(V.kind == "montage") & (V.montage == montage)]
    for f in K.LOG_FEATURES:
        d = d.assign(**{f: np.log(d[f].where(d[f] > 0))})
    return d.groupby("subject")[K.F17].mean()


# ----------------------------------------------------------------------------- C-b one CV repeat
def cv_predict(X, y, method, model, seed, n_splits=5):
    pred = np.zeros(len(y))
    sels = []
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        Xtr = X.iloc[tr].reset_index(drop=True)
        sel = SEL.select(method, Xtr, y[tr], np.arange(len(tr)), "reg")
        if sel == "pca":
            pca = SEL.PCAReference().fit(X.iloc[tr].values)
            A, B = pca.transform(X.iloc[tr].values), pca.transform(X.iloc[te].values)
            sel = list(K.F17)
        else:
            A, B = X.iloc[tr][sel].values, X.iloc[te][sel].values
        if model == "Ridge":
            sc = StandardScaler().fit(A)
            m = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(sc.transform(A), y[tr])
            pred[te] = m.predict(sc.transform(B))
        else:
            pred[te] = RandomForestRegressor(300, min_samples_leaf=3, random_state=seed, n_jobs=1).fit(A, y[tr]).predict(B)
        sels.append(sel)
    return pred, sels


def run_method(X, y, method, model, repeats, n_perm, seed=K.SEED):
    rhos, maes, sel_all = [], [], []
    for r in range(repeats):
        p, sels = cv_predict(X, y, method, model, seed + r)
        rhos.append(stats.spearmanr(p, y).statistic)
        maes.append(mean_absolute_error(y, p))
        sel_all += sels
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        yp = rng.permutation(y)
        p, _ = cv_predict(X, yp, method, model, seed)
        null.append(stats.spearmanr(p, yp).statistic)
    null = np.asarray(null)
    p_perm = (1 + np.sum(null >= rhos[0])) / (1 + n_perm) if n_perm else np.nan
    Z = np.array([[f in s for f in K.F17] for s in sel_all], int)
    stab = M.nogueira_stability(Z) if method not in ("all", "panel", "pca") else dict(phi=np.nan)
    return {"method": method, "model": model, "rho_mean": float(np.mean(rhos)),
            "rho_lo": float(np.percentile(rhos, 2.5)), "rho_hi": float(np.percentile(rhos, 97.5)),
            "mae_mean": float(np.mean(maes)), "p_perm": float(p_perm), "n_selected": float(Z.sum(1).mean()),
            "nogueira": stab["phi"], "selection_freq": dict(zip(K.F17, Z.mean(0).round(3)))}


SCORE_LABELS = {"nihss": "NIHSS", "mbi": "MBI", "mrs": "mRS"}


def plot_association(assoc, targets, path):
    """Forest plot of Spearman rho per feature and score (can be re-drawn from association.csv)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for i, t in enumerate(targets):
        d = assoc[assoc.target == t].set_index("feature").loc[K.F17]
        ax.errorbar(d.rho, np.arange(17) + (i - (len(targets) - 1) / 2) * 0.22, xerr=[d.rho - d.lo, d.hi - d.rho],
                    fmt="o", ms=4, capsize=2, label=SCORE_LABELS.get(t, t))
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks(range(17))
    ax.set_yticklabels([K.LABELS[f] for f in K.F17], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Spearman rho with clinical score (bootstrap 95% CI)")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--methods", nargs="+", default=SEL.METHODS)
    ap.add_argument("--inspect", action="store_true", help="only print the detected clinical columns")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    clin, cols, src = clinical_table()
    if args.inspect:
        print(clin.describe(include="all").round(2).to_string())
        return
    X = patient_features()
    clin = clin.reindex(X.index)
    targets = [t for t in SEVERITY if t in clin and clin[t].notna().sum() >= max(10, 0.6 * len(X))]
    (OUT / "clinical_columns.json").write_text(json.dumps({"source": Path(str(src)).name, "detected": cols, "targets": targets,
                                                           "n_patients": int(len(X))}, indent=2))
    if not targets:
        raise SystemExit(f"no severity score (NIHSS, MBI/Barthel, mRS) found; detected columns: {cols}. "
                         "Add the column names to PATTERNS in this script.")
    print("severity targets:", targets, clin[targets].describe().round(2).to_string())
    covars = [c for c in ("age", "days") if c in clin and clin[c].notna().all()]

    # ---------------- C-a association
    rows = []
    for t in targets:
        ok = clin[t].notna()
        for f in K.F17:
            x, y = X.loc[ok, f].values, clin.loc[ok, t].values
            rho, p = stats.spearmanr(x, y)
            lo, hi = M.bootstrap_ci(lambda a, b: stats.spearmanr(a, b).statistic, x, y)
            pr, pp = M.partial_spearman(x, y, [clin.loc[ok, c].values for c in covars]) if covars else (np.nan, np.nan)
            rows.append({"target": t, "feature": f, "n": int(ok.sum()), "rho": rho, "lo": lo, "hi": hi, "p": p,
                         "partial_rho": pr, "partial_p": pp})
    assoc = pd.DataFrame(rows)
    assoc["p_fdr"] = assoc.groupby("target").p.transform(lambda p: T.p_adjust(p.values, "fdr_bh"))
    assoc["partial_p_fdr"] = assoc.groupby("target").partial_p.transform(
        lambda p: T.p_adjust(p.values, "fdr_bh") if p.notna().all() else np.nan)
    assoc.to_csv(OUT / "association.csv", index=False)

    # ---------------- C-b prediction of the primary score
    primary = targets[0]
    ok = clin[primary].notna()
    Xp, yp = X.loc[ok].reset_index(drop=True), clin.loc[ok, primary].values.astype(float)
    jobs = [(m, mdl) for m in args.methods for mdl in ("Ridge", "Random forest")]
    res = Parallel(n_jobs=args.n_jobs, verbose=5)(delayed(run_method)(Xp, yp, m, mdl, args.repeats, args.n_perm)
                                                 for m, mdl in jobs)
    pred = pd.DataFrame([{k: v for k, v in r.items() if k != "selection_freq"} for r in res])
    pred["target"] = primary
    pred["mae_baseline_mean"] = float(np.mean(np.abs(yp - yp.mean())))
    pred.to_csv(OUT / "prediction.csv", index=False)
    pd.DataFrame([{"method": r["method"], "model": r["model"], **r["selection_freq"]} for r in res]).to_csv(
        OUT / "prediction_selection.csv", index=False)
    if covars:   # reference: ridge on the clinical covariates only (age, days since stroke)
        A = clin.loc[ok, covars].values.astype(float)
        rhos, maes = [], []
        for r in range(args.repeats):
            p = np.zeros(len(yp))
            for tr, te in KFold(5, shuffle=True, random_state=K.SEED + r).split(A):
                sc = StandardScaler().fit(A[tr])
                p[te] = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(sc.transform(A[tr]), yp[tr]).predict(sc.transform(A[te]))
            rhos.append(stats.spearmanr(p, yp).statistic)
            maes.append(mean_absolute_error(yp, p))
        (OUT / "covariate_reference.json").write_text(json.dumps(
            {"covariates": covars, "rho_mean": float(np.mean(rhos)), "mae_mean": float(np.mean(maes))}, indent=2))

    # ---------------- C-c montage robustness of the strongest associations
    top = assoc[assoc.target == primary].reindex(
        assoc[assoc.target == primary].rho.abs().sort_values(ascending=False).index).head(4).feature.tolist()
    mrows = []
    for m in K.MONTAGES:
        try:
            Xm = patient_features(m).reindex(X.index)
        except Exception:
            continue
        for f in top:
            okm = ok & Xm[f].notna()
            if okm.sum() >= 10:
                rho, p = stats.spearmanr(Xm.loc[okm, f], clin.loc[okm, primary])
                mrows.append({"montage": m, "feature": f, "rho": rho, "p": p})
    pd.DataFrame(mrows).to_csv(OUT / "montage_association.csv", index=False)

    # ---------------- LaTeX rows
    a = assoc[assoc.target == primary].set_index("feature")
    lines = []
    for f in K.F17:
        r = a.loc[f]
        part = "--" if pd.isna(r.partial_rho) else f"{r.partial_rho:+.2f} ({r.partial_p_fdr:.2f})"
        lines.append(f"{K.LABELS[f]} & {r.rho:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}] & {r.p:.3f} & {r.p_fdr:.2f} & {part}\\\\")
    (OUT / "table_C_association.tex").write_text("\n".join(lines) + "\n")
    lines = []
    for m in args.methods:
        rr = pred[pred.method == m].set_index("model")
        cells = []
        for mdl in ("Ridge", "Random forest"):
            r = rr.loc[mdl]
            cells += [f"{r.rho_mean:+.2f} [{r.rho_lo:+.2f}, {r.rho_hi:+.2f}]", f"{r.mae_mean:.2f}", f"{r.p_perm:.2f}"]
        stab = "--" if pd.isna(rr.loc["Ridge", "nogueira"]) else f"{rr.loc['Ridge', 'nogueira']:.2f}"
        lines.append(" & ".join([SEL.METHOD_LABELS[m], f"{rr.loc['Ridge', 'n_selected']:.1f}", stab] + cells) + r"\\")
    (OUT / "table_C_prediction.tex").write_text("\n".join(lines) + "\n")

    plot_association(assoc, targets, OUT / "fig_association.png")
    print(assoc.sort_values("p").head(10).round(3).to_string())
    print(pred[["method", "model", "rho_mean", "mae_mean", "p_perm", "n_selected"]].round(3).to_string())


if __name__ == "__main__":
    main()
