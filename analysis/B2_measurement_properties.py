"""Study B, step 2: reliability, epoch length, electrode sparsity and channel dropout.

  B-a reliability : one-way random ICC(1,1) across a patient's 40 trials, ICC(1,40) of the patient mean,
                    and the trials (and seconds) needed for ICC >= 0.75 / 0.90 (Spearman-Brown)
  B-b epoch length: ICC per window length and agreement (Lin's CCC) of patient means with the 8-s value
  B-c montage     : agreement of patient means with the 29-channel value, redundancy structure per
                    montage (|r| >= 0.6 pairs, similarity to the 29-channel matrix, corr/VIF chain result)
  B-d dropout     : agreement of patient means after random loss of 2, 4 or 8 electrodes
Features in common.LOG_FEATURES are analysed on the log scale. CIs: F-based (ICC), patient bootstrap (CCC).

Outputs (MDPI_RESULTS/studyB/): reliability.csv, length_agreement.csv, montage_agreement.csv,
montage_redundancy.csv, dropout_agreement.csv, table_B.tex, fig_reliability_length.png,
fig_montage_ccc.png, fig_dropout.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as K
import measures as M
import selection as S

OUT = K.OUT / "studyB"


def transform(df):
    df = df.copy()
    for f in K.LOG_FEATURES:
        v = df[f].where(df[f] > 0)
        df[f] = np.log(v)
    return df


def patient_means(d):
    return d.groupby("subject")[K.F17].mean()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    V = transform(pd.read_csv(K.WORK / "features_variants.csv"))
    L = V[V.kind == "length"]
    ref4 = L[L.length_s == 4]

    # ---------------- B-a/B-b reliability by window length
    rel = []
    for Ls, d in L.groupby("length_s"):
        for f in K.F17:
            Y = d.pivot(index="subject", columns="trial", values=f).values
            r = M.icc_oneway(Y)
            rel.append({"feature": f, "length_s": Ls, **r,
                        "trials_for_075": M.spearman_brown_k(r["icc1"], 0.75),
                        "trials_for_090": M.spearman_brown_k(r["icc1"], 0.90)})
    rel = pd.DataFrame(rel)
    rel["seconds_for_090"] = rel.trials_for_090 * rel.length_s
    rel.to_csv(OUT / "reliability.csv", index=False)

    ref8 = patient_means(L[L.length_s == max(K.EPOCH_LENGTHS_S)])
    lag = []
    for Ls, d in L.groupby("length_s"):
        pm = patient_means(d).loc[ref8.index]
        for f in K.F17:
            lo, hi = M.bootstrap_ci(M.ccc, pm[f].values, ref8[f].values)
            lag.append({"feature": f, "length_s": Ls, "ccc": M.ccc(pm[f], ref8[f]), "lo": lo, "hi": hi})
    pd.DataFrame(lag).to_csv(OUT / "length_agreement.csv", index=False)

    # ---------------- B-c montage agreement and redundancy
    pm29 = patient_means(ref4)
    mon_rows, red_rows = [], []
    tables = {"M29": ref4}
    tables.update({m: d for m, d in V[V.kind == "montage"].groupby("montage")})
    r29 = ref4[K.F17].corr().values
    iu = np.triu_indices(17, 1)
    keep29, _ = S.fit_redundancy(ref4[K.F17])
    for m in K.MONTAGES:
        if m not in tables:
            continue
        d = tables[m]
        pm = patient_means(d).reindex(pm29.index)
        for f in K.F17:
            ok = pm[f].notna() & pm29[f].notna()
            if ok.sum() < 5:
                mon_rows.append({"montage": m, "feature": f, "ccc": np.nan, "lo": np.nan, "hi": np.nan, "icc1": np.nan})
                continue
            lo, hi = M.bootstrap_ci(M.ccc, pm[f][ok].values, pm29[f][ok].values)
            Y = d.pivot(index="subject", columns="trial", values=f).values
            mon_rows.append({"montage": m, "feature": f, "ccc": M.ccc(pm[f][ok], pm29[f][ok]), "lo": lo, "hi": hi,
                             "icc1": M.icc_oneway(Y)["icc1"]})
        X = d[K.F17].dropna(axis=1, how="all")
        r = X.corr().reindex(index=K.F17, columns=K.F17).values
        keep, rep = S.fit_redundancy(X)
        ok = ~np.isnan(r[iu]) & ~np.isnan(r29[iu])
        red_rows.append({"montage": m, "n_channels": len(K.MONTAGES[m]) if K.MONTAGES[m] else 29,
                         "n_features_available": X.shape[1],
                         "pairs_r06": int((np.abs(r[iu][ok]) >= 0.6).sum()),
                         "similarity_to_M29": float(np.corrcoef(r[iu][ok], r29[iu][ok])[0, 1]) if m != "M29" else 1.0,
                         "n_retained": len(keep), "retained": ";".join(keep),
                         "jaccard_vs_M29": len(set(keep) & set(keep29)) / len(set(keep) | set(keep29))})
    mon = pd.DataFrame(mon_rows)
    mon.to_csv(OUT / "montage_agreement.csv", index=False)
    pd.DataFrame(red_rows).to_csv(OUT / "montage_redundancy.csv", index=False)

    # ---------------- B-d channel dropout
    Dd = V[V.kind == "dropout"]
    intact = patient_means(ref4[ref4.trial < K.DROPOUT_TRIALS])
    drop_rows = []
    for (dn, rep), d in Dd.groupby(["dropout_n", "rep"]):
        pm = patient_means(d).loc[intact.index]
        for f in K.F17:
            drop_rows.append({"dropout_n": dn, "rep": rep, "feature": f, "ccc": M.ccc(pm[f], intact[f])})
    drop = pd.DataFrame(drop_rows).groupby(["dropout_n", "feature"]).ccc.agg(["mean", "min"]).reset_index()
    drop.to_csv(OUT / "dropout_agreement.csv", index=False)

    # ---------------- LaTeX summary table (one row per feature)
    r4 = rel[rel.length_s == 4].set_index("feature")
    la = pd.DataFrame(lag).query("length_s == 2").set_index("feature")
    mc = mon.pivot(index="feature", columns="montage", values="ccc")
    dr = drop[drop.dropout_n == max(K.DROPOUT_LEVELS)].set_index("feature")["mean"]
    lines = []
    fmt = lambda v: "--" if pd.isna(v) else f"{v:.2f}"
    for f in K.F17:
        k90 = r4.loc[f, "trials_for_090"]
        lines.append(" & ".join([K.LABELS[f], f"{r4.loc[f, 'icc1']:.2f} [{r4.loc[f, 'lo']:.2f}, {r4.loc[f, 'hi']:.2f}]",
                                 f"{r4.loc[f, 'icck']:.2f}", "--" if not np.isfinite(k90) else f"{k90:.1f}",
                                 fmt(la.loc[f, "ccc"])] +
                                [fmt(mc.loc[f].get(m)) for m in ("M19", "M8", "M4C", "M2")] + [fmt(dr.get(f))]) + r"\\")
    (OUT / "table_B.tex").write_text("\n".join(lines) + "\n")

    # ---------------- figures
    piv = rel.pivot(index="feature", columns="length_s", values="icc1").loc[K.F17]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    im = axes[0].imshow(piv.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    axes[0].set_xticks(range(piv.shape[1]))
    axes[0].set_xticklabels([f"{c:g} s" for c in piv.columns])
    axes[0].set_yticks(range(17))
    axes[0].set_yticklabels([K.LABELS[f] for f in piv.index], fontsize=8)
    for i in range(17):
        for j in range(piv.shape[1]):
            axes[0].text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=6,
                         color="white" if piv.values[i, j] < 0.5 else "black")
    axes[0].set_title("ICC(1,1) across trials", fontsize=10)
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    sec = rel.pivot(index="feature", columns="length_s", values="seconds_for_090").loc[K.F17]
    for Ls in sec.columns:
        axes[1].scatter(sec[Ls].clip(upper=1e4), range(17), label=f"{Ls:g}-s windows", s=18)
    axes[1].set_xscale("log")
    axes[1].set_yticks(range(17))
    axes[1].set_yticklabels([K.LABELS[f] for f in sec.index], fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("recording time for ICC of the mean >= 0.90 (s, log)")
    axes[1].legend(fontsize=7)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_reliability_length.png", dpi=300)
    plt.close(fig)

    order = [m for m in K.MONTAGES if m in mc.columns and m != "M29"]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    vals = mc.loc[K.F17, order].values
    im = ax.imshow(vals, cmap="magma", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_yticks(range(17))
    ax.set_yticklabels([K.LABELS[f] for f in K.F17], fontsize=8)
    for i in range(17):
        for j in range(len(order)):
            if not np.isnan(vals[i, j]):
                ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if vals[i, j] < 0.5 else "black")
    ax.set_title("Agreement with the 29-channel value (Lin's CCC)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT / "fig_montage_ccc.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for fam, fs in K.FAMILIES.items():
        d = drop[drop.feature.isin(fs)].groupby("dropout_n")["mean"].mean()
        ax.plot(d.index, d.values, marker="o", label=fam)
    ax.set_xlabel("electrodes lost (of 29)")
    ax.set_ylabel("CCC with intact montage (family mean)")
    ax.set_xticks(list(K.DROPOUT_LEVELS))
    ax.legend(fontsize=7, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_dropout.png", dpi=300)
    plt.close(fig)
    print(rel[rel.length_s == 4][["feature", "icc1", "icck", "trials_for_090"]].round(3).to_string())
    print(pd.DataFrame(red_rows)[["montage", "pairs_r06", "similarity_to_M29", "n_retained", "jaccard_vs_M29"]])


if __name__ == "__main__":
    main()
