"""Study B, step 4 (review revision): redundancy at the patient level and within patients.

Correlations over all trials pool 47 patients x 40 trials and conflate stable between-patient differences with
within-patient co-fluctuation. This script reports, for the 4-s, 29-channel features (log scale as before):
  pooled   : correlations over all trials (as in the first version of the paper)
  between  : correlations of the 47 patient means (patient discrimination)
  within   : correlations of the trial values after subtracting each patient's mean (within-patient change)
and, for each level, the number of pairs with |r| >= 0.6, the subset kept by the correlation + VIF chain, a
patient bootstrap (resampling patients) of how often each feature is kept, and the agreement between levels.

  python B4_hierarchy.py [--boot 500]
Outputs (MDPI_RESULTS/studyB/): hierarchy_corr_{pooled,between,within}.csv, hierarchy_summary.csv,
hierarchy_selection_frequency.csv, table_B4.tex
"""
import argparse

import numpy as np
import pandas as pd

import common as K
import selection as S

OUT = K.OUT / "studyB"


def level_tables(d):
    """d: trial-level feature table (subject, trial, 17 log-transformed features)."""
    pooled = d[K.F17]
    means = d.groupby("subject")[K.F17].mean()
    within = d[K.F17] - d.groupby("subject")[K.F17].transform("mean")
    return {"pooled": pooled, "between": means, "within": within}


def summarise(tabs, thr=0.6):
    iu = np.triu_indices(len(K.F17), 1)
    rows, keeps, corrs = [], {}, {}
    for lev, X in tabs.items():
        r = X.corr().reindex(index=K.F17, columns=K.F17)
        corrs[lev] = r
        keep, rep = S.fit_redundancy(X)
        keeps[lev] = keep
        v = r.values[iu]
        rows.append({"level": lev, "n_rows": len(X), "pairs_r06": int((np.abs(v) >= thr).sum()),
                     "median_abs_r": float(np.nanmedian(np.abs(v))), "n_retained": len(keep),
                     "retained": ";".join(keep)})
    out = pd.DataFrame(rows)
    sim = {}
    for a in tabs:
        for b in tabs:
            if a < b:
                ok = ~np.isnan(corrs[a].values[iu]) & ~np.isnan(corrs[b].values[iu])
                sim[f"corr_similarity_{a}_{b}"] = float(np.corrcoef(corrs[a].values[iu][ok], corrs[b].values[iu][ok])[0, 1])
                sa, sb = set(keeps[a]), set(keeps[b])
                sim[f"jaccard_{a}_{b}"] = len(sa & sb) / len(sa | sb)
    return out, sim, corrs, keeps


def bootstrap_selection(d, n_boot, seed=K.SEED):
    rng = np.random.default_rng(seed)
    subs = d.subject.unique()
    groups = {s: g for s, g in d.groupby("subject")}
    cnt = {lev: pd.Series(0, index=K.F17, dtype=float) for lev in ("pooled", "between", "within")}
    n_ret = {lev: [] for lev in cnt}
    for _ in range(n_boot):
        pick = rng.choice(subs, len(subs), replace=True)
        b = pd.concat([groups[s].assign(subject=i) for i, s in enumerate(pick)], ignore_index=True)
        for lev, X in level_tables(b).items():
            try:
                keep, _ = S.fit_redundancy(X)
            except Exception:
                continue
            cnt[lev][keep] += 1
            n_ret[lev].append(len(keep))
    freq = pd.DataFrame({lev: cnt[lev] / n_boot for lev in cnt})
    return freq, {lev: (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for lev, v in n_ret.items()}


def run(V, n_boot, write=True):
    d = V[(V.kind == "length") & (V.length_s == 4)][["subject", "trial"] + K.F17].copy()
    for f in K.LOG_FEATURES:
        d[f] = np.log(d[f].where(d[f] > 0))
    d = d.dropna()
    tabs = level_tables(d)
    summ, sim, corrs, keeps = summarise(tabs)
    freq, nret = bootstrap_selection(d, n_boot)
    summ["n_retained_boot_mean"] = [nret[l][0] for l in summ.level]
    summ["n_retained_boot_lo"] = [nret[l][1] for l in summ.level]
    summ["n_retained_boot_hi"] = [nret[l][2] for l in summ.level]
    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        for lev, r in corrs.items():
            r.to_csv(OUT / f"hierarchy_corr_{lev}.csv")
        summ.to_csv(OUT / "hierarchy_summary.csv", index=False)
        freq.to_csv(OUT / "hierarchy_selection_frequency.csv")
        pd.Series(sim).to_csv(OUT / "hierarchy_agreement.csv", header=["value"])
        lines = [" & ".join([K.LABELS[f]] + [f"{freq.loc[f, l]:.2f}" for l in ("pooled", "between", "within")]) + r"\\" for f in K.F17]
        (OUT / "table_B4.tex").write_text("\n".join(lines) + "\n")
    return summ, sim, freq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=500)
    args = ap.parse_args()
    K.ensure_dirs()
    V = pd.read_csv(K.WORK / "features_variants.csv")
    summ, sim, freq = run(V, args.boot)
    print(summ.to_string()); print(sim)


if __name__ == "__main__":
    main()
