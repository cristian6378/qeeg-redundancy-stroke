"""Step 0 (run once, before step 1): where does motor imagery start inside each 8-s trial?

Prints, so you can paste it back:
  1. every row of the raw .mat array with its channel name (EDF / .locs) and amplitude spread,
     so non-EEG rows (reference, EOG, event marker, unnamed) can be identified;
  2. the timing of the experimental events, from BIDS *_events.tsv files if the download has them
     (value 1 = instruction, 2 = motor imagery, 3 = break; MOABB, moabb.datasets.Liu2024), and from
     any marker-like row inside the .mat file;
  3. a suggested MI_WINDOW_S for config.py.
"""

import numpy as np
import pandas as pd

import config as C
import data as D


def main():
    root = C.DATA_ROOT
    files = D.subject_files(root)
    s0 = min(files)
    raw, lab = D._load_mat(files[s0])
    n_tr, n_rows, n_s = raw.shape
    print(f"\n=== sub-{s0:02d}: rawdata {raw.shape} (trials x rows x samples) = {n_s / C.FS_RAW:.1f} s per trial")

    names, src = None, None
    edfs = sorted(root.rglob("*.edf"))
    if edfs:
        names, src = D._edf_labels(edfs[0]), edfs[0].name
    locs = sorted(root.rglob("*.locs"))
    locs_names = D._locs_labels(locs[0]) if locs else []
    print(f"channel names from: {src}  ({len(names) if names else 0} labels); .locs has {len(locs_names)} labels")

    print("\nrow | EDF label | .locs label | std | n unique values (first 3 trials) | looks like")
    for r in range(n_rows):
        x = raw[:3, r, :]
        nu = len(np.unique(np.round(x, 6)))
        kind = "flat" if x.std() < 1e-6 else ("MARKER (few levels)" if nu <= 20 else "signal")
        e = names[r] if names and r < len(names) else "-"
        l = locs_names[r] if r < len(locs_names) else "-"
        print(f"{r:3d} | {e!s:>8} | {l!s:>8} | {x.std():10.3f} | {nu:6d} | {kind}")

    # ---- marker-like rows: report where the levels change inside trials
    print("\n=== marker-like rows inside the .mat (time of each level change, seconds from trial start)")
    found = False
    for r in range(n_rows):
        x = raw[:, r, :]
        if x.std() < 1e-6 or len(np.unique(np.round(x[:3], 6))) > 20:
            continue
        found = True
        for t in range(min(4, n_tr)):
            ch = np.flatnonzero(np.diff(x[t]) != 0) + 1
            vals = [f"{i / C.FS_RAW:.2f}s->{x[t, i]:g}" for i in ch[:10]]
            print(f"row {r}, trial {t} (label {lab[t]}): first value {x[t, 0]:g}; changes: {vals}")
    if not found:
        print("none")

    # ---- BIDS events files
    ev = sorted(p for p in root.rglob("*.tsv") if "event" in p.name.lower())
    print(f"\n=== events files found: {len(ev)}")
    rows = []
    for p in ev[:60]:
        try:
            df = pd.read_csv(p, sep="\t")
        except Exception as e:
            print("could not read", p.name, e); continue
        if p == ev[0]:
            print(f"{p.name}: columns {list(df.columns)}\n{df.head(8).to_string()}")
        if {"onset", "value"} <= set(df.columns):
            df = df.sort_values("onset").reset_index(drop=True)
            for i in range(len(df) - 1):
                if df.value[i] == 1 and df.value[i + 1] == 2:
                    mi_onset = df.onset[i + 1] - df.onset[i]
                    nxt = df.onset[i + 2] - df.onset[i] if i + 2 < len(df) else np.nan
                    rows.append({"file": p.name, "mi_onset_after_instruction_s": mi_onset,
                                 "next_event_after_instruction_s": nxt,
                                 "mi_duration_s": df.duration[i + 1] if "duration" in df else np.nan})
    if rows:
        R = pd.DataFrame(rows)
        print("\nmotor-imagery onset relative to the instruction (= trial start), over all trials:")
        print(R.describe().round(3).to_string())
        on = float(np.median(R.mi_onset_after_instruction_s))
        dur = float(np.nanmedian(R.mi_duration_s)) if R.mi_duration_s.notna().any() else 4.0
        dur = dur if dur > 0 else 4.0
        print(f"\nSUGGESTED: MI_WINDOW_S = ({on:.1f}, {min(on + dur, n_s / C.FS_RAW):.1f})   (current: {C.MI_WINDOW_S})")
    else:
        print("no usable events table; use the marker rows above, or the dataset's code.zip, to place the window")


if __name__ == "__main__":
    main()
