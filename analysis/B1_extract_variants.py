"""Study B, step 1: compute the 17 features under the measurement conditions of a wearable.

For each of the 47 patients and each of the 40 trials (8 s, Study-1 filter chain at 500 Hz):
  length   : first 1, 2, 4 and 8 s of the trial, all 29 channels
  montage  : first 4 s, electrode subsets M19, M8, M4F, M4C, M2 (common.MONTAGES)
  dropout  : first 4 s, 2, 4 or 8 of the 29 channels removed at random (3 draws; first 20 trials)
Absolute band powers (sums of squared DFT magnitudes) are divided by N^2, i.e. expressed as mean
power per sample in the band (Parseval), so that windows of different length are on the same
scale; ratios, asymmetry, distribution, complexity and connectivity are scale-free.
No task labels are used anywhere in Study B.

  python B1_extract_variants.py [--n-jobs -1]
Output: MDPI_WORK/features_variants.csv
"""
import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import common as K
import config as C
import data as D
import features as F

BANDS = list(C.BANDS)


def feats(x, names, n_samples):
    out = F.trial_features(x, C.FS_RAW, F.homologous_index_pairs(names))
    for b in BANDS:
        out[b] = out[b] / n_samples ** 2       # Parseval: sum |X_k|^2 = N sum x^2 -> mean power in the band
    return out


def one_subject(s, path, keep, names):
    raw, _ = D._load_mat(path)
    x = D.qeeg_filter(raw[:, keep, :])               # filter the full 8-s trial
    pos = {n: i for i, n in enumerate(names)}
    rows = []
    n_full = x.shape[-1]
    for t in range(x.shape[0]):
        for L in K.EPOCH_LENGTHS_S:
            n = min(int(L * C.FS_RAW), n_full)
            rows.append({"subject": s, "trial": t, "kind": "length", "length_s": L, "montage": "M29",
                         "dropout_n": 0, "rep": 0, **feats(x[t, :, :n], names, n)})
        n4 = int(4 * C.FS_RAW)
        for m, chans in K.MONTAGES.items():
            if chans is None:
                continue
            idx = [pos[c] for c in chans if c in pos]
            if len(idx) < 2:
                continue
            rows.append({"subject": s, "trial": t, "kind": "montage", "length_s": 4, "montage": m,
                         "dropout_n": 0, "rep": 0, **feats(x[t][idx, :n4], [names[i] for i in idx], n4)})
        if t < K.DROPOUT_TRIALS:
            for d in K.DROPOUT_LEVELS:
                for r in range(K.DROPOUT_REPS):
                    rng = np.random.default_rng([K.SEED, s, t, d, r])
                    idx = np.sort(rng.choice(len(names), len(names) - d, replace=False))
                    rows.append({"subject": s, "trial": t, "kind": "dropout", "length_s": 4, "montage": "M29",
                                 "dropout_n": d, "rep": r, **feats(x[t][idx, :n4], [names[i] for i in idx], n4)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()
    K.ensure_dirs()
    files = D.subject_files(C.DATA_ROOT)
    raw0, _ = D._load_mat(files[min(files)])
    keep, names, _ = D.resolve_channels(C.DATA_ROOT, raw0.shape[1], raw0)
    print(f"{len(files)} patients, {len(names)} channels; montages:",
          {m: (len(names) if c is None else len([n for n in c if n in names])) for m, c in K.MONTAGES.items()})
    res = Parallel(n_jobs=args.n_jobs, verbose=5)(delayed(one_subject)(s, p, keep, names) for s, p in files.items())
    df = pd.DataFrame([r for rows in res for r in rows])
    df.to_csv(K.WORK / "features_variants.csv", index=False)
    print(df.groupby("kind").size(), "\n->", K.WORK / "features_variants.csv")


if __name__ == "__main__":
    main()
