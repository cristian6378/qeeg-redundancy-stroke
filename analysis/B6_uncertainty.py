"""Study B, step 6 (review revision): 95% patient-bootstrap intervals for the quantities that were reported as
point values in the first revision.

  (a) task-matched repeatability (4-s imagery window unless stated): ICC(1,1) and the DIFFERENCES
      second half - first half, left - right imagery, after - before the quality screen,
      rest 2 s - imagery 2 s, instruction 2 s - imagery 2 s, imagery 4 s - first 4 s of the trial (primary window)
  (b) montage redundancy: pairs with |r| >= 0.6, similarity of the correlation matrix to M29 and Jaccard
      overlap of the corr/VIF subset with that of M29
  (c) random electrode loss: CCC of the patient means with the intact montage, per number of lost electrodes,
      pooled over the three random draws

Every interval resamples PATIENTS with replacement (the same resample for all quantities of one replicate, so
differences are paired) and is a percentile interval.

  python B6_uncertainty.py [--boot 1000] [--boot-redundancy 200]
Needs MDPI_WORK/features_taskmatched.csv (B3) and MDPI_WORK/features_variants.csv (B1).
Outputs (MDPI_RESULTS/studyB/): uncertainty_taskmatched.csv, uncertainty_montage_redundancy.csv,
uncertainty_dropout.csv
"""
import argparse

import numpy as np
import pandas as pd

import common as K
import measures as M
import selection as S

OUT = K.OUT / "studyB"


def transform(df):
    df = df.copy()
    for f in K.LOG_FEATURES:
        df[f] = np.log(df[f].where(df[f] > 0))
    return df


def mat(d, f, subjects):
    """subjects x trials matrix (trial rank within subject), rows ordered as `subjects`, complete columns only."""
    d = d.assign(k=d.groupby("subject").cumcount())
    Y = d.pivot(index="subject", columns="k", values=f).reindex(subjects)
    return Y.dropna(axis=1, how="any").values


def pct(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if len(v) else (np.nan, np.nan)


# ------------------------------------------------------------------ (a)
def taskmatched(T, V, n_boot, seed=K.SEED):
    subs = np.array(sorted(T.subject.unique()))
    sel = lambda ph, L: T[(T.phase == ph) & (T.length_s == L)]
    img4 = sel("imagery", 4)
    conds = {
        "img4": img4,
        "img4_first": img4[img4.trial < 20], "img4_second": img4[img4.trial >= 20],
        "img4_left": img4[img4.hand == 1], "img4_right": img4[img4.hand == 2],
        "img4_qc": img4[~img4.qc_flag.astype(bool)] if "qc_flag" in img4 else None,
        "img2": sel("imagery", 2), "rest2": sel("rest", 2), "instr2": sel("instruction", 2),
    }
    if V is not None:
        conds["first4"] = V[(V.kind == "length") & (V.length_s == 4)]
    conds = {k: v for k, v in conds.items() if v is not None and len(v)}
    contrasts = {"second_minus_first": ("img4_second", "img4_first"), "left_minus_right": ("img4_left", "img4_right"),
                 "qc_minus_all": ("img4_qc", "img4"), "rest2_minus_img2": ("rest2", "img2"),
                 "instr2_minus_img2": ("instr2", "img2"), "img4_minus_first4": ("img4", "first4")}
    contrasts = {k: v for k, v in contrasts.items() if v[0] in conds and v[1] in conds}
    rng = np.random.default_rng(seed)
    idx = [rng.integers(0, len(subs), len(subs)) for _ in range(n_boot)]
    rows = []
    for f in K.F17:
        Y = {c: mat(d, f, subs) for c, d in conds.items()}
        point = {c: M.icc_oneway(y)["icc1"] for c, y in Y.items()}
        boots = {c: np.array([M.icc_oneway(y[i])["icc1"] for i in idx]) for c, y in Y.items()}
        for c in conds:
            lo, hi = pct(boots[c])
            rows.append({"feature": f, "quantity": f"icc1_{c}", "estimate": point[c], "lo": lo, "hi": hi})
        for name, (a, b) in contrasts.items():
            lo, hi = pct(boots[a] - boots[b])
            rows.append({"feature": f, "quantity": name, "estimate": point[a] - point[b], "lo": lo, "hi": hi,
                         "excludes_zero": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ (b)
def _red(X, r29, keep29, iu):
    X = X.dropna(axis=1, how="all")
    r = X.corr().reindex(index=K.F17, columns=K.F17).values
    keep, _ = S.fit_redundancy(X)
    ok = ~np.isnan(r[iu]) & ~np.isnan(r29[iu])
    return (int((np.abs(r[iu][ok]) >= 0.6).sum()), float(np.corrcoef(r[iu][ok], r29[iu][ok])[0, 1]),
            len(set(keep) & set(keep29)) / len(set(keep) | set(keep29)), len(keep))


def montage_redundancy(V, n_boot, seed=K.SEED):
    ref = V[(V.kind == "length") & (V.length_s == 4)]
    tabs = {"M29": ref, **{m: d for m, d in V[V.kind == "montage"].groupby("montage")}}
    groups = {m: {s: g for s, g in d.groupby("subject")} for m, d in tabs.items()}
    subs = np.array(sorted(ref.subject.unique()))
    iu = np.triu_indices(17, 1)
    rng = np.random.default_rng(seed)
    res = {m: [] for m in tabs}
    point = {}
    r29 = ref[K.F17].corr().values
    keep29, _ = S.fit_redundancy(ref[K.F17])
    for m, d in tabs.items():
        point[m] = _red(d[K.F17], r29, keep29, iu)
    for _ in range(n_boot):
        pick = rng.choice(subs, len(subs), replace=True)
        b29 = pd.concat([groups["M29"][s] for s in pick])[K.F17]
        rb29 = b29.corr().values
        kb29, _ = S.fit_redundancy(b29)
        for m in tabs:
            bm = pd.concat([groups[m][s] for s in pick if s in groups[m]])[K.F17]
            try:
                res[m].append(_red(bm, rb29, kb29, iu))
            except Exception:
                continue
    rows = []
    for m in tabs:
        a = np.array(res[m], float)
        for j, q in enumerate(["pairs_r06", "similarity_to_M29", "jaccard_vs_M29", "n_retained"]):
            lo, hi = pct(a[:, j])
            rows.append({"montage": m, "quantity": q, "estimate": point[m][j], "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ (c)
def dropout(V, n_boot, seed=K.SEED):
    ref = V[(V.kind == "length") & (V.length_s == 4)]
    intact = ref[ref.trial < K.DROPOUT_TRIALS].groupby("subject")[K.F17].mean()
    D = V[V.kind == "dropout"]
    rng = np.random.default_rng(seed)
    subs = intact.index.values
    idx = [rng.integers(0, len(subs), len(subs)) for _ in range(n_boot)]
    rows = []
    for dn, d in D.groupby("dropout_n"):
        pms = [g.groupby("subject")[K.F17].mean().reindex(subs) for _, g in d.groupby("rep")]
        for f in K.F17:
            est = np.mean([M.ccc(pm[f].values, intact[f].values) for pm in pms])
            bs = [np.mean([M.ccc(pm[f].values[i], intact[f].values[i]) for pm in pms]) for i in idx]
            lo, hi = pct(bs)
            rows.append({"dropout_n": dn, "feature": f, "ccc_mean_over_draws": est, "lo": lo, "hi": hi,
                         "min_over_draws": min(M.ccc(pm[f].values, intact[f].values) for pm in pms)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--boot-redundancy", type=int, default=200)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    V = transform(pd.read_csv(K.WORK / "features_variants.csv"))
    T = transform(pd.read_csv(K.WORK / "features_taskmatched.csv"))
    print("(a) task-matched repeatability ...", flush=True)
    a = taskmatched(T, V, args.boot); a.to_csv(OUT / "uncertainty_taskmatched.csv", index=False)
    print(a[a.quantity.str.contains("minus")].groupby("quantity").excludes_zero.sum())
    print("(b) montage redundancy ...", flush=True)
    b = montage_redundancy(V, args.boot_redundancy); b.to_csv(OUT / "uncertainty_montage_redundancy.csv", index=False)
    print(b.round(2).to_string())
    print("(c) electrode loss ...", flush=True)
    c = dropout(V, args.boot); c.to_csv(OUT / "uncertainty_dropout.csv", index=False)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
