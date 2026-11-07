"""Loading, channel resolution and preprocessing for the Liu et al. (2024) stroke MI dataset."""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sps
from scipy.io import loadmat

import config as C


# ============================================================================ .mat loading
def _load_mat(path: Path):
    """Return (rawdata [trials, rows, samples], labels [trials]) from one subject file."""
    try:
        m = loadmat(path, squeeze_me=False)
        eeg = m["eeg"]
        raw = np.asarray(eeg["rawdata"][0, 0], dtype=np.float64)
        lab = np.asarray(eeg["label"][0, 0]).ravel()
    except NotImplementedError:  # MATLAB v7.3 -> HDF5
        import h5py
        with h5py.File(path, "r") as f:
            raw = np.array(f["eeg"]["rawdata"]).T.astype(np.float64)
            lab = np.array(f["eeg"]["label"]).ravel()
    return raw, lab.astype(int)


def subject_files(root: Path):
    """Find one .mat per subject anywhere under root (tolerates extra folder levels from unzipping).

    The subject number is read from 'sub-XX' / 'subXX' in the file or folder name."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Data folder does not exist: {root.resolve()}")
    out = {}
    for p in sorted(root.rglob("*.mat")):
        m = None
        for part in (p.stem, *[q.name for q in p.parents]):
            m = re.search(r"sub[-_]?0*(\d+)", part, flags=re.I)
            if m:
                break
        if m:
            s = int(m.group(1))
            if 1 <= s <= C.N_SUBJECTS and s not in out and s not in C.EXCLUDE_SUBJECTS:
                out[s] = p
    if not out:
        zips = sorted(p.name for p in root.rglob("*.zip"))
        hint = (f" Found zip files that still need unzipping: {zips}." if zips else
                " Download sourcedata.zip from figshare and unzip it into this folder.")
        raise FileNotFoundError(f"No sub-XX .mat files anywhere under {root.resolve()}.{hint}")
    print(f"[data] subject files found under: {out[min(out)].parent.parent}")
    print(f"[data] {len(out)} subjects used; excluded duplicate recordings: {list(C.EXCLUDE_SUBJECTS)}")
    expected = C.N_SUBJECTS - len(C.EXCLUDE_SUBJECTS)
    if len(out) < expected:
        missing = [i for i in range(1, C.N_SUBJECTS + 1) if i not in out and i not in C.EXCLUDE_SUBJECTS]
        warnings.warn(f"only {len(out)} of {expected} subjects found; missing: {missing}")
    return out


# ============================================================================ channel names
def _edf_labels(path: Path):
    """Minimal EDF header reader (no extra dependency)."""
    with open(path, "rb") as f:
        hdr = f.read(256)
        ns = int(hdr[252:256].decode("ascii").strip())
        labels = f.read(16 * ns).decode("latin-1")
    return [labels[i * 16:(i + 1) * 16].strip() for i in range(ns)]


def _locs_labels(path: Path):
    names = []
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.split()
        if len(parts) >= 4 and re.match(r"^-?\d+", parts[0]):
            names.append(parts[-1])
    return names


def _clean(name: str) -> str:
    name = re.sub(r"^(EEG|EOG)[\s\-_]*", "", name.strip(), flags=re.I)
    name = re.sub(r"[\s\-_]*(REF|LE|AVG)$", "", name, flags=re.I)
    return name.upper()


def resolve_channels(root: Path, n_rows: int, raw_example: np.ndarray):
    """Map rawdata rows to channel names and choose the EEG rows.

    Order of preference for names: an EDF header from edffile.zip, then
    channel_location.locs from code.zip. The mapping is written to work/channels.json and
    MUST be checked once by eye (see README, step 2).
    """
    names, source = None, None
    edfs = sorted(root.rglob("*.edf"))
    if edfs:
        names, source = _edf_labels(edfs[0]), str(edfs[0])
    else:
        locs = sorted(list(root.rglob("*.locs")) + list(root.rglob("*.ced")))
        if locs:
            names, source = _locs_labels(locs[0]), str(locs[0])

    rows_std = raw_example.std(axis=(0, 2))
    if names is None:
        warnings.warn("No .edf/.locs found: using the thesis fallback rows and NO channel names. "
                      "pDBSI and the foundation-model channel embedding need names, so add "
                      "edffile.zip or code.zip to DATA_ROOT before the final run.")
        keep = [r for r in range(n_rows) if r != C.FALLBACK_DROP_ROW][:C.FALLBACK_N_EEG]
        return keep, [f"CH{r}" for r in keep], {"source": None}

    if len(names) > n_rows:
        names = names[:n_rows]
    if len(names) < n_rows:
        warnings.warn(f"{len(names)} names for {n_rows} rows: rows {len(names)}..{n_rows - 1} "
                      "are treated as non-EEG (e.g., event marker).")
    clean = [_clean(n) for n in names]
    keep = []
    for r, n in enumerate(clean):
        if not n or not re.match(r"^[A-Z]{1,3}[0-9Z]{1,2}$", n):   # unnamed / non-10-20 label -> not EEG
            continue
        if any(h in n for h in C.NON_EEG_NAME_HINTS):
            continue
        if n in C.REFERENCE_CHANNELS:
            continue
        if rows_std[r] < 1e-6:          # flat row
            continue
        keep.append(r)
    info = {"source": source,
            "all_rows": [{"row": r, "name": (clean[r] if r < len(clean) else None),
                          "std": float(rows_std[r]), "kept": r in keep} for r in range(n_rows)]}
    return keep, [clean[r] for r in keep], info


# ============================================================================ labels
def hemiplegia_labels(root: Path):
    """Return ({subject: 0 = left hemiplegia | 1 = right hemiplegia}, source) or (None, None).

    Sources, in order: a user-made hemiplegia.csv (columns subject,side with side L/R), then
    participants.tsv from the figshare deposit. Only columns that describe the HEMIPLEGIC
    (paretic/affected) side are used; a lesion-side column is NOT used, because a left-hemisphere
    lesion produces RIGHT hemiplegia. Counts are checked against the dataset paper (27 L / 23 R).
    """
    cands = list(root.rglob("hemiplegia.csv")) + list(root.rglob("participants.tsv")) + \
        list(root.rglob("participants.csv"))
    for p in cands:
        df = pd.read_csv(p, sep="\t" if p.suffix == ".tsv" else None, engine="python")
        cols = {c.lower().strip(): c for c in df.columns}
        print(f"[labels] {p.name}: columns = {list(df.columns)}")
        side_col = None
        for key in ("hemipleg", "paretic", "affected", "side"):
            side_col = next((cols[c] for c in cols if key in c and "lesion" not in c), None)
            if side_col:
                break
        sub_col = next((cols[c] for c in cols if c in ("participant_id", "subject", "id", "sub")),
                       next((cols[c] for c in cols if "sub" in c or "participant" in c), None))
        if side_col is None or sub_col is None:
            print(f"[labels] {p.name}: no hemiplegia-side column recognised")
            continue
        out = {}
        for s, v in zip(df[sub_col], df[side_col]):
            m = re.search(r"(\d+)", str(s))
            v = str(v).strip().lower()
            if not m:
                continue
            if v.startswith("l") or v in ("左", "左侧"):
                out[int(m.group(1))] = 0
            elif v.startswith("r") or v in ("右", "右侧"):
                out[int(m.group(1))] = 1
        if len(out) >= 10:
            n_l = sum(1 for k, v in out.items() if v == 0 and k not in C.EXCLUDE_SUBJECTS)
            n_r = sum(1 for k, v in out.items() if v == 1 and k not in C.EXCLUDE_SUBJECTS)
            print(f"[labels] using column '{side_col}' of {p.name}: {n_l} left / {n_r} right "
                  f"(paper, all 50: 27 left / 23 right)")
            return out, f"{p} [{side_col}]"
    return None, None


# ============================================================================ preprocessing
def crop_window(x, fs, window_s):
    a, b = int(round(window_s[0] * fs)), int(round(window_s[1] * fs))
    return x[..., a:b]


def qeeg_filter(x, fs=C.FS_RAW, f=C.QEEG_FILTERS):
    """Study-1 filter chain: 4th-order Butterworth HP 0.1 Hz, notch 50 Hz (Q=30), LP 40 Hz;
    zero-phase, per channel. Applied to the full 8-s trial before cropping (less edge effect)."""
    b, a = sps.butter(f["order"], f["hp"] / (fs / 2), "highpass")
    x = sps.filtfilt(b, a, x, axis=-1)
    b, a = sps.iirnotch(f["notch"], f["notch_q"], fs)
    x = sps.filtfilt(b, a, x, axis=-1)
    b, a = sps.butter(f["order"], f["lp"] / (fs / 2), "lowpass")
    return sps.filtfilt(b, a, x, axis=-1)


def fm_preprocess(x, fs=C.FS_RAW):
    """ST-EEGFormer input: band-pass 0.5-63 Hz + 50 Hz notch, resample to 128 Hz,
    per-trial per-channel z-score (model card). Applied to the full trial, then cropped."""
    sos = sps.butter(4, [C.FM_BANDPASS[0] / (fs / 2), C.FM_BANDPASS[1] / (fs / 2)],
                     "bandpass", output="sos")
    x = sps.sosfiltfilt(sos, x, axis=-1)
    b, a = sps.iirnotch(C.FM_NOTCH, 30.0, fs)
    x = sps.filtfilt(b, a, x, axis=-1)
    x = sps.resample_poly(x, C.FS_FM, fs, axis=-1)
    return x


def zscore_channels(x, eps=1e-8):
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    return (x - mu) / (sd + eps)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
