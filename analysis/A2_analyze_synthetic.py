"""Study A, step 2: accuracy, parsimony, stability, redundancy and cost of each selection strategy.

Units: outer folds for descriptive means (25), synthetic subjects for inference (n = 40; per-subject
balanced accuracy averaged over repeats). Tests: Friedman + Nemenyi across strategies (per
classifier) and two-sided Wilcoxon tests of every strategy against 'All 17', Holm-corrected.

Outputs (MDPI_RESULTS/studyA/): summary.csv, per_subject.csv, tests_vs_all.csv, omnibus.csv,
selection_frequency.csv, table_A.tex, fig_pareto.png, fig_selection_frequency.png, cd_*.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score, f1_score

import common as K
import fs_methods as SEL
import measures as M
import selection as S
import stats_tools as T

OUT = K.OUT / "studyA"


def modal_subset(sel_rows):
    c = pd.Series([tuple(sorted(r["selected"])) for r in sel_rows]).value_counts()
    return list(c.index[0]), int(c.iloc[0])


def main():
    pred = pd.read_csv(OUT / "predictions.csv")
    sels = [json.loads(line) for line in open(OUT / "selections.jsonl")]
    data = pd.read_csv(OUT / "synthetic_used.csv")
    cost = K.cost_table()
    methods = [m for m in SEL.METHODS if m in pred.method.unique()]

    # ---------------- accuracy per fold and per subject
    fold = pred.groupby(["method", "classifier", "repeat", "fold"]).apply(
        lambda d: pd.Series({"macro_f1": f1_score(d.y_true, d.y_pred, average="macro"),
                             "bal_acc": balanced_accuracy_score(d.y_true, d.y_pred)}), include_groups=False).reset_index()
    subj = pred.assign(ok=(pred.y_true == pred.y_pred).astype(float)).groupby(
        ["method", "classifier", "repeat", "subject"]).ok.mean().groupby(["method", "classifier", "subject"]).mean()
    subj = subj.rename("accuracy").reset_index()
    subj.to_csv(OUT / "per_subject.csv", index=False)

    # ---------------- selection: size, stability, redundancy, cost, coverage
    rows, freq = [], []
    Xall = data[K.F17]
    for m in methods:
        sr = [s for s in sels if s["method"] == m]
        Z = np.array([[f in s["selected"] for f in K.F17] for s in sr], int)
        stab = M.nogueira_stability(Z) if m not in ("all", "panel", "pca") else dict(phi=np.nan, lo=np.nan, hi=np.nan)
        sub, n_modal = modal_subset(sr)
        red_r = np.nan
        max_vif = np.nan
        if m != "pca" and len(sub) > 1:
            r = Xall[list(sub)].corr().abs().values
            red_r = float(np.median(r[np.triu_indices(len(sub), 1)]))
            max_vif = float(S.vif(Xall[list(sub)].dropna()).max())
        rows.append({"method": m, "n_inputs_mean": np.mean([s["n_inputs"] for s in sr]),
                     "n_inputs_min": min(s["n_inputs"] for s in sr), "n_inputs_max": max(s["n_inputs"] for s in sr),
                     "nogueira": stab["phi"], "nogueira_lo": stab["lo"], "nogueira_hi": stab["hi"],
                     "jaccard": M.mean_jaccard([s["selected"] for s in sr]) if m != "pca" else np.nan,
                     "modal_subset": ";".join(sub), "modal_count": n_modal, "n_runs": len(sr),
                     "families_covered": len({K.FAMILY_OF[f] for f in sub}),
                     "median_abs_r": red_r, "max_vif": max_vif,
                     "cost_ms": np.mean([K.subset_cost(s["selected"], cost) for s in sr]),
                     "selection_s": np.mean([s["selection_s"] for s in sr])})
        freq.append(pd.Series(Z.mean(0), index=K.F17, name=m))
    sel_df = pd.DataFrame(rows)
    freq = pd.DataFrame(freq)
    freq.to_csv(OUT / "selection_frequency.csv")

    acc = fold.groupby(["method", "classifier"]).agg(f1_mean=("macro_f1", "mean"), f1_sd=("macro_f1", "std"),
                                                     ba_mean=("bal_acc", "mean"), ba_sd=("bal_acc", "std")).reset_index()
    summary = acc.merge(sel_df, on="method")
    summary.to_csv(OUT / "summary.csv", index=False)

    # ---------------- tests
    tests, omni = [], []
    for c in pred.classifier.unique():
        W = subj[subj.classifier == c].pivot(index="subject", columns="method", values="accuracy")[methods]
        fr = T.friedman(W.values)
        _, ranks, cd = T.nemenyi(W.values, methods)
        omni.append({"classifier": c, **fr, "nemenyi_cd": cd, **{f"rank_{m}": ranks[m] for m in methods}})
        T.cd_diagram({SEL.METHOD_LABELS[m]: r for m, r in ranks.items()}, cd,
                     f"Study A - {c} (n={len(W)} subjects, Friedman p={fr['p']:.2g})",
                     OUT / f"cd_{c.split()[0].lower()}.png")
        for m in methods:
            if m == "all":
                continue
            d = W[m] - W["all"]
            p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
            tests.append({"classifier": c, "method": m, "mean_diff_vs_all": d.mean(), "p": p})
    tests = pd.DataFrame(tests)
    tests["p_holm"] = tests.groupby("classifier").p.transform(lambda p: T.p_adjust(p.values, "holm"))
    tests.to_csv(OUT / "tests_vs_all.csv", index=False)
    pd.DataFrame(omni).to_csv(OUT / "omnibus.csv", index=False)

    # ---------------- LaTeX rows (one per strategy; both classifiers)
    lines = []
    for m in methods:
        r = summary[summary.method == m]
        lr = r[r.classifier == "Logistic Regression"].iloc[0]
        gb = r[r.classifier == "LightGBM"].iloc[0]
        t_lr = tests[(tests.method == m) & (tests.classifier == "Logistic Regression")]
        t_gb = tests[(tests.method == m) & (tests.classifier == "LightGBM")]
        star = lambda t: "" if t.empty or t.p_holm.iloc[0] >= 0.05 else "$^{*}$"
        stab = "--" if np.isnan(lr.nogueira) else f"{lr.nogueira:.2f}"
        lines.append(" & ".join([SEL.METHOD_LABELS[m], f"{lr.n_inputs_mean:.1f}", f"${lr.f1_mean:.3f}\\pm{lr.f1_sd:.3f}${star(t_lr)}",
                                 f"${gb.f1_mean:.3f}\\pm{gb.f1_sd:.3f}${star(t_gb)}", stab,
                                 "--" if np.isnan(lr.max_vif) else f"{lr.max_vif:.1f}", f"{lr.cost_ms:.1f}"]) + r"\\")
    (OUT / "table_A.tex").write_text("\n".join(lines) + "\n")

    # ---------------- figures
    # accuracy-cost-stability panel: one row per strategy (sorted by cost), three aligned columns
    lr = summary[summary.classifier == "Logistic Regression"].set_index("method")
    gb = summary[summary.classifier == "LightGBM"].set_index("method")
    order = lr.loc[methods].sort_values("cost_ms", ascending=False).index.tolist()
    yy = np.arange(len(order))
    lo = 0.94
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.2), sharey=True, gridspec_kw={"width_ratios": [2.2, 1.6, 1]},
                             layout="constrained")
    ax = axes[0]
    for d, mk, lab, dy in ((lr, "o", "Logistic regression", -0.13), (gb, "^", "LightGBM", 0.13)):
        v = d.loc[order, "f1_mean"].values
        e = d.loc[order, "f1_sd"].values
        col = "#1f4e79" if mk == "o" else "#c55a11"
        ok = v >= lo
        ax.errorbar(v[ok], (yy + dy)[ok], xerr=e[ok], fmt=mk, ms=5, capsize=2, label=lab, color=col)
        for y, val in zip(yy[~ok], v[~ok]):             # off-scale (stability selection): print the value
            ax.plot(lo + 0.0006, y + dy, marker="<", color=col, ms=5, clip_on=False)
            ax.annotate(f"{val:.3f} (off scale)", (lo + 0.0006, y + dy), xytext=(6, -3),
                        textcoords="offset points", fontsize=7, color=col)
    ax.set_xlim(lo, 1.003)
    ax.set_xlabel("macro-F1 (mean \u00b1 SD, 25 outer folds)")
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    ax.set_yticks(yy)
    ax.set_yticklabels([SEL.METHOD_LABELS[m] for m in order], fontsize=8)
    ax = axes[1]
    phi = lr.loc[order, "nogueira"].values
    cmap = plt.get_cmap("viridis")
    ax.barh(yy, lr.loc[order, "cost_ms"].values, color=[("#bbbbbb" if np.isnan(f) else cmap(f)) for f in phi],
            edgecolor="k", linewidth=0.3)
    for y, c in zip(yy, lr.loc[order, "cost_ms"].values):
        ax.text(c * 1.15, y, f"{c:.0f}" if c >= 10 else f"{c:.1f}", va="center", fontsize=7)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 3000)
    ax.set_xlabel("extraction cost per 4-s window (ms, log)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes.tolist(), location="right", shrink=0.8, pad=0.02)
    cb.set_label("selection stability \u03a6 (grey: fixed set)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax = axes[2]
    ax.barh(yy, lr.loc[order, "n_inputs_mean"].values, color="#9dc3e6", edgecolor="k", linewidth=0.3)
    ax.set_xlabel("inputs (mean)")
    ax.set_xlim(0, 18)
    for a_ in axes:
        a_.spines[["top", "right"]].set_visible(False)
        a_.grid(axis="x", lw=0.3, alpha=0.5)
    fig.savefig(OUT / "fig_pareto.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    f2 = freq.drop(index=[m for m in ("all", "pca") if m in freq.index])
    im = ax.imshow(f2.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(f2)))
    ax.set_yticklabels([SEL.METHOD_LABELS[m] for m in f2.index], fontsize=8)
    ax.set_xticks(range(17))
    ax.set_xticklabels([K.LABELS[f] for f in K.F17], rotation=60, ha="right", fontsize=8)
    for i in range(f2.shape[0]):
        for j in range(17):
            if f2.values[i, j] > 0:
                ax.text(j, i, f"{f2.values[i, j]:.1f}".lstrip("0"), ha="center", va="center", fontsize=5.5,
                        color="white" if f2.values[i, j] > 0.6 else "black")
    fig.colorbar(im, ax=ax, label="selection frequency over outer folds")
    fig.tight_layout()
    fig.savefig(OUT / "fig_selection_frequency.png", dpi=300)
    plt.close(fig)
    print(summary[["method", "classifier", "f1_mean", "n_inputs_mean", "nogueira", "cost_ms"]].round(3).to_string())


if __name__ == "__main__":
    main()
