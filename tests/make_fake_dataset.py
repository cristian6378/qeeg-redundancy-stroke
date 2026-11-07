"""Simulated stand-in for figshare 21679035 (Liu et al., 2024), used by the smoke test.

Same layout as the real deposit: data/sourcedata/sub-XX/*.mat with eeg.rawdata (40 trials x rows x
4000 samples at 500 Hz) and eeg.label (1 = left, 2 = right hand), data/code/channel_location.locs
and data/participants.tsv. The EEG is noise plus a 10-Hz rhythm with contralateral attenuation
during the first 4 s, and a mild hemispheric asymmetry for left-hemiplegia patients.

  python make_fake_dataset.py OUT_DIR [--subjects 10] [--montage compact|full]
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.io import savemat

FULL = ["FP1", "FP2", "FZ", "F3", "F4", "F7", "F8", "FCZ", "FC3", "FC4", "FT7", "FT8", "CZ", "C3", "C4", "T3", "T4",
        "CPZ", "CP3", "CP4", "TP7", "TP8", "PZ", "P3", "P4", "T5", "T6", "OZ", "O1", "O2"]
COMPACT = ["FP1", "FP2", "F3", "F4", "FC3", "FC4", "CZ", "C3", "C4", "CPZ", "CP3", "CP4", "P3", "P4", "O1", "O2"]
EXTRA = ["HEOG", "VEOG", "TRIGGER"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--subjects", type=int, default=10)
    ap.add_argument("--montage", choices=["compact", "full"], default="compact")
    a = ap.parse_args()
    root, rng = Path(a.out), np.random.default_rng(0)
    names = (FULL if a.montage == "full" else COMPACT) + EXTRA
    idx = {n: i for i, n in enumerate(names)}
    n_eeg = len(names) - len(EXTRA)
    (root / "code").mkdir(parents=True, exist_ok=True)
    with open(root / "code" / "channel_location.locs", "w") as f:
        for i, n in enumerate(names):
            f.write(f"{i + 1}\t{rng.uniform(-180, 180):.1f}\t0.4\t{n}\n")
    t = np.arange(4000) / 500
    rows = ["participant_id\tGender\tAge\themiplegia\tTime after stroke (days)\tNIHSS\tMBI\tmRS"]
    left_hemisphere = [idx[n] for n in ("F3", "FC3", "C3", "CP3", "P3") if n in idx]
    for s in range(1, a.subjects + 1):
        side = "L" if s % 2 else "R"
        nihss = int(rng.integers(1, 15))                                # simulated severity: more slowing
        rows.append(f"sub-{s:02d}\t{'M' if s % 3 else 'F'}\t{50 + s}\t{side}\t{int(rng.integers(1, 30))}\t{nihss}"
                    f"\t{max(0, 100 - 6 * nihss + int(rng.integers(-10, 10)))}\t{min(5, nihss // 3)}")
        lab = np.repeat([1, 2], 20)
        rng.shuffle(lab)
        X = rng.normal(0, 10, (40, len(names), 4000))
        X[:, :n_eeg] += 8 * np.sin(2 * np.pi * 10 * t[None, None] + rng.uniform(0, 2 * np.pi, (40, n_eeg, 1)))
        X[:, :n_eeg] += 0.8 * nihss * np.sin(2 * np.pi * 2 * t[None, None] + rng.uniform(0, 2 * np.pi, (40, n_eeg, 1)))
        for k in range(40):
            X[k, idx["C4" if lab[k] == 1 else "C3"], :2000] *= 0.6     # contralateral mu attenuation
            if side == "L":
                X[k, left_hemisphere] *= 1.25                          # lesion asymmetry
        X[:, idx["CPZ"]] = 0.0                                          # recording reference
        X[:, idx["TRIGGER"]] = 0.0
        X[:, idx["TRIGGER"], 0] = 1
        d = root / "sourcedata" / f"sub-{s:02d}"
        d.mkdir(parents=True, exist_ok=True)
        savemat(d / f"sub-{s:02d}_task-motor-imagery_eeg.mat",
                {"eeg": {"rawdata": X.astype(np.float32), "label": lab.reshape(-1, 1)}})
    (root / "participants.tsv").write_text("\n".join(rows) + "\n")
    print(f"simulated {a.subjects} patients, {n_eeg} EEG channels -> {root}")


if __name__ == "__main__":
    main()
