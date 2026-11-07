"""Study B, step 3 (review revision): task-matched windows and within-session repeatability by task phase.

The eight-second trials of the Liu et al. dataset are not homogeneous: an instruction interval precedes
the motor-imagery (MI) interval, which is followed by rest. Windows that all start at 0 s therefore mix
task phases in different proportions. Here every window lies inside one phase:

  instruction : [0, L)                       L = 1, 2      (only if L <= MI onset)
  imagery     : [onset, onset+L)             L = 1, 2, 4   nested windows inside the imagery interval
  imagery_late: [onset+4-L, onset+4)         L = 1, 2      (position control inside the imagery interval)
  rest        : [onset+4, onset+4+L)         L = 1, 2      (only if it fits in the trial)
  full8       : [0, 8)                       the mixture used in the first version of the paper

ICC(1,1) across a patient's trials is computed per phase, per window length and per hand (all / left / right
imagery trials), together with the patient-bootstrap interval of the recording time needed for a
patient-mean reliability of 0.90 (Spearman-Brown) and a trial-order check (first vs second half of the
session). These are WITHIN-SESSION repeatabilities, not test-retest reliabilities.

IMPORTANT: --mi-onset must be confirmed with  python ../pipeline/00_check_timing.py  (events: 1 instruction,
2 motor imagery, 3 break). The default (2.0 s) is the usual description of the dataset and is not verified here.

  python B3_taskmatched.py [--mi-onset 2.0] [--mi-length 4.0] [--n-jobs -1] [--boot 200]
Outputs: MDPI_WORK/features_taskmatched.csv; MDPI_RESULTS/studyB/reliability_taskmatched.csv,
         reliability_taskmatched_qc.csv (after the quality screen), qc_summary.csv, table_B3.tex, fig_taskmatched.png
"""
import argparse

import numpy as np
import pandas as pd

import common as K
import measures as M

OUT = K.OUT / "studyB"
TRIAL_S = 8.0


def window_spec(onset, mi_len, trial_s=TRIAL_S):
    """List of (phase, length_s, start_s) with every window entirely inside its phase."""
    spec = [("full8", trial_s, 0.0)]
    for L in (1, 2):
        if L <= onset:
            spec.append(("instruction", L, 0.0))
    for L in (1, 2, 4):
        if L <= mi_len:
            spec.append(("imagery", L, onset))
    for L in (1, 2):
        if L < mi_len:
            spec.append(("imagery_late", L, onset + mi_len - L))
    for L in (1, 2):
        if onset + mi_len + L <= trial_s:
            spec.append(("rest", L, onset + mi_len))
    return spec


def one_subject(s, path, keep, names, onset, mi_len):
    import config as C
    import data as D
    import B1_extract_variants as B1
    raw, lab = D._load_mat(path)
    x = D.qeeg_filter(raw[:, keep, :])
    # objective, unit-free quality screen on the full trial: a trial is flagged if any channel's peak-to-peak
    # amplitude exceeds 5x that channel's median over the patient's trials, or a channel is flat (SD < 1% of median)
    ptp = np.ptp(x, axis=-1)                                   # trials x channels
    sd = x.std(axis=-1)
    ratio = (ptp / np.median(ptp, axis=0, keepdims=True)).max(axis=1)
    flat = (sd < 0.01 * np.median(sd, axis=0, keepdims=True)).any(axis=1)
    rows = []
    for t in range(x.shape[0]):
        qc = {"qc_ptp_ratio": float(ratio[t]), "qc_flat": bool(flat[t]), "qc_flag": bool(ratio[t] > 5 or flat[t])}
        for phase, L, start in window_spec(onset, mi_len):
            a, b = int(round(start * C.FS_RAW)), int(round((start + L) * C.FS_RAW))
            b = min(b, x.shape[-1])
            rows.append({"subject": s, "trial": t, "hand": int(lab[t]) if t < len(lab) else -1,
                         "phase": phase, "length_s": L, "start_s": start, **qc,
                         **B1.feats(x[t, :, a:b], names, b - a)})
    return rows


def _prep(df):
    d = df.copy()
    for f in K.LOG_FEATURES:
        v = d[f].where(d[f] > 0)
        d[f] = np.log(v)
    return d


def _matrix(d, f):
    """subjects x trials matrix of one feature within the rows of d (trials ranked inside each subject)."""
    d = d.assign(k=d.groupby("subject").cumcount())
    return d.pivot(index="subject", columns="k", values=f).dropna(axis=1, how="any").values


def icc_table(df, n_boot=200, seed=K.SEED):
    """ICC per (phase, length, hand, feature) with bootstrap CI of the seconds needed for reliability 0.90."""
    d = _prep(df)
    rng = np.random.default_rng(seed)
    rows = []
    hands = {"all": None, "left": 1, "right": 2}
    for (phase, L), g in d.groupby(["phase", "length_s"]):
        for hname, h in hands.items():
            gg = g if h is None else g[g.hand == h]
            if gg.groupby("subject").size().min() < 4:
                continue
            subs = gg.subject.unique()
            for f in K.F17:
                Y = _matrix(gg, f)
                if Y.shape[0] < 5 or Y.shape[1] < 3:
                    continue
                r = M.icc_oneway(Y)
                k90 = M.spearman_brown_k(r["icc1"], 0.90)
                # patient bootstrap of the recording time
                secs = []
                if hname == "all" and n_boot:
                    for _ in range(n_boot):
                        idx = rng.integers(0, Y.shape[0], Y.shape[0])
                        rb = M.icc_oneway(Y[idx])["icc1"]
                        kb = M.spearman_brown_k(rb, 0.90)
                        secs.append(kb * L)
                half = Y.shape[1] // 2
                r1, r2 = M.icc_oneway(Y[:, :half])["icc1"], M.icc_oneway(Y[:, half:2 * half])["icc1"]
                rows.append({"phase": phase, "length_s": L, "hand": hname, "feature": f,
                             "n_subjects": Y.shape[0], "n_trials": Y.shape[1], **r,
                             "trials_for_090": k90, "seconds_for_090": k90 * L,
                             "seconds_lo": np.nanpercentile(np.where(np.isfinite(secs), secs, np.nan), 2.5) if secs else np.nan,
                             "seconds_hi": np.nanpercentile(np.where(np.isfinite(secs), secs, np.nan), 97.5) if secs else np.nan,
                             "share_boot_infinite": float(np.mean(~np.isfinite(secs))) if secs else np.nan,
                             "icc1_first_half": r1, "icc1_second_half": r2})
    return pd.DataFrame(rows)


def write_outputs(tab):
    OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / "reliability_taskmatched.csv", index=False)
    a = tab[tab.hand == "all"]
    lines = []
    fmt = lambda v: "--" if pd.isna(v) else f"{v:.2f}"
    key = [("full8", 8), ("imagery", 1), ("imagery", 2), ("imagery", 4), ("imagery_late", 2), ("instruction", 2), ("rest", 2)]
    for f in K.F17:
        cells = []
        for ph, L in key:
            r = a[(a.phase == ph) & (a.length_s == L) & (a.feature == f)]
            cells.append(fmt(r.icc1.iloc[0]) if len(r) else "--")
        lines.append(" & ".join([K.LABELS[f]] + cells) + r"\\")
    (OUT / "table_B3.tex").write_text("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    piv = a.assign(c=a.phase + " " + a.length_s.astype(int).astype(str) + " s").pivot(index="feature", columns="c", values="icc1").reindex(K.F17)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    im = ax.imshow(piv.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(17)); ax.set_yticklabels([K.LABELS[f] for f in piv.index], fontsize=8)
    fig.colorbar(im, label="ICC(1,1)"); fig.tight_layout(); fig.savefig(OUT / "fig_taskmatched.png", dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mi-onset", type=float, default=2.0)
    ap.add_argument("--mi-length", type=float, default=4.0)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--skip-extract", action="store_true")
    args = ap.parse_args()
    K.ensure_dirs()
    p = K.WORK / "features_taskmatched.csv"
    if not args.skip_extract:
        from joblib import Parallel, delayed
        import config as C
        import data as D
        files = D.subject_files(C.DATA_ROOT)
        raw0, _ = D._load_mat(files[min(files)])
        keep, names, _ = D.resolve_channels(C.DATA_ROOT, raw0.shape[1], raw0)
        print(f"WARNING: using MI onset {args.mi_onset} s, imagery length {args.mi_length} s; "
              "confirm with pipeline/00_check_timing.py")
        res = Parallel(n_jobs=args.n_jobs, verbose=5)(delayed(one_subject)(s, f, keep, names, args.mi_onset, args.mi_length)
                                                     for s, f in files.items())
        pd.DataFrame([r for rows in res for r in rows]).to_csv(p, index=False)
    df = pd.read_csv(p)
    tab = icc_table(df, n_boot=args.boot)
    write_outputs(tab)
    if "qc_flag" in df:
        q = df[df.phase == "full8"].groupby("subject").qc_flag.agg(["size", "sum"]).rename(columns={"size": "trials", "sum": "flagged"})
        q["retained"] = q.trials - q.flagged
        q.to_csv(OUT / "qc_summary.csv")
        print(f"quality screen: {int(q.flagged.sum())} of {int(q.trials.sum())} trials flagged; "
              f"retained per patient median {q.retained.median():.0f} (min {q.retained.min():.0f})")
        icc_table(df[~df.qc_flag.astype(bool)], n_boot=0).to_csv(OUT / "reliability_taskmatched_qc.csv", index=False)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
