r"""Shared configuration for Study 2 (real stroke EEG).

Repository layout (paths are resolved from THIS file, so the scripts run from any folder):

    stroke-eeg-qeeg-vs-foundation-models/     <- PROJECT_DIR (repository root)
        pipeline/                             <- this code (config.py lives here)
        data/                                 <- figshare 21679035, unzipped (or pipeline/data/)
            sourcedata/sub-01/*.mat ... sub-50/*.mat      (sourcedata.zip)
            edffile/...  code/...                          (edffile.zip, code.zip: channel names)
            participants.tsv                               (affected side, Task Side)
        STEEGFormer/                          <- git clone https://github.com/LiuyinYang1101/STEEGFormer.git
        checkpoints/small/checkpoint-300.pth                (release ST-EEGFormer-small)
        checkpoints/base/checkpoint-288.pth                 (optional)
        checkpoints/large/large_weights_only_196.pth        (optional)
        work/  results/                       <- created by the pipeline

Every location can be overridden with an environment variable: STROKE_EEG_ROOT (data),
STROKE_EEG_WORK, STROKE_EEG_RESULTS, STEEGFORMER_REPO, STEEGFORMER_SMALL / _BASE / _LARGE.
Check the resolved paths with:  python config.py
"""
from pathlib import Path
import os

# ----------------------------------------------------------------------------- paths
CODE_DIR = Path(__file__).resolve().parent          # <repo>/pipeline
PROJECT_DIR = CODE_DIR.parent                        # <repo>


def _path(env_var, *candidates):
    """Environment variable if it points to something that exists; otherwise the first candidate
    that exists (else the first one). A stale or placeholder variable is reported and ignored."""
    val = os.environ.get(env_var, "").strip().strip('"')
    if val:
        p = Path(val).expanduser()
        if p.exists():
            return p.resolve()
        print(f"[config] ignoring {env_var}={val!r} (does not exist); using the default location.")
    for c in candidates:
        if Path(c).exists():
            return Path(c).resolve()
    return Path(candidates[0]).resolve()


def _both(*rel):
    """Look in the repository root first, then inside pipeline/."""
    return PROJECT_DIR.joinpath(*rel), CODE_DIR.joinpath(*rel)


_data_default = next((p for p in _both("data") if (p / "sourcedata").exists()), PROJECT_DIR / "data")
DATA_ROOT = _path("STROKE_EEG_ROOT", _data_default)
# output folders may not exist yet, so an environment variable is used as given (Colab, other disks)
WORK_DIR = Path(os.environ.get("STROKE_EEG_WORK") or PROJECT_DIR / "work").expanduser().resolve()
RESULTS_DIR = Path(os.environ.get("STROKE_EEG_RESULTS") or PROJECT_DIR / "results").expanduser().resolve()
STEEGFORMER_REPO = _path("STEEGFORMER_REPO", *_both("STEEGFormer"))
STEEGFORMER_CKPT = {  # github.com/LiuyinYang1101/STEEGFormer/releases (exact release file names)
    "small": _path("STEEGFORMER_SMALL", *_both("checkpoints", "small", "checkpoint-300.pth")),
    "base": _path("STEEGFORMER_BASE", *_both("checkpoints", "base", "checkpoint-288.pth")),
    "large": _path("STEEGFORMER_LARGE", *_both("checkpoints", "large", "large_weights_only_196.pth")),
}

# ----------------------------------------------------------------------------- recording
FS_RAW = 500                 # Hz (Liu et al., 2024)
N_TRIALS_PER_SUBJECT = 40
N_SUBJECTS = 50
# sub-05, sub-15 and sub-49 are not independent recordings: in sourcedata they equal sub-04,
# sub-11 and sub-47 plus uniform noise (documented by MOABB, moabb.datasets.Liu2024). Keeping them
# would put one patient's EEG on both sides of a cross-patient split, so they are excluded.
EXCLUDE_SUBJECTS = (5, 15, 49)
# rawdata is trials x 33 x 4000. Non-EEG rows (reference, EOG, event marker) are dropped by
# name using the .locs file; this fallback is used only if no channel file is found and
# reproduces the thesis notebook (drop row 17 = reference, keep the first 29 of the rest).
FALLBACK_DROP_ROW = 17
FALLBACK_N_EEG = 29
NON_EEG_NAME_HINTS = ("EOG", "HEO", "VEO", "ECG", "EMG", "TRIG", "MARK", "EVENT", "STATUS")
REFERENCE_CHANNELS = ("CPZ",)   # recording reference (flat or near-flat); ground FPz is not stored

# Motor-imagery window inside the 8-s trial (seconds from trial start). The dataset paper's
# baseline decoders use 0-4 s; CHECK against the stimulus timing in code.zip before the final run.
MI_WINDOW_S = (0.0, 4.0)

# ----------------------------------------------------------------------------- qEEG branch
QEEG_FILTERS = dict(hp=0.1, notch=50.0, notch_q=30.0, lp=40.0, order=4)   # identical to Study 1
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 18)}
# Homologous pairs are resolved BY NAME (only pairs present in the montage are used).
HOMOLOGOUS_PAIRS = [("FP1", "FP2"), ("F3", "F4"), ("F7", "F8"), ("FC3", "FC4"), ("FT7", "FT8"),
                    ("C3", "C4"), ("T3", "T4"), ("T7", "T8"), ("CP3", "CP4"), ("TP7", "TP8"),
                    ("P3", "P4"), ("T5", "T6"), ("P7", "P8"), ("O1", "O2")]
FEATURES_17 = ["delta", "theta", "alpha", "beta", "delta_alpha", "theta_alpha", "dt_ab",
               "kurtosis", "skewness", "sample_entropy", "higuchi_fd",
               "weighted_phase_lag_index", "magnitude_squared_coherence",
               "pdbsi_delta", "pdbsi_theta", "pdbsi_alpha", "pdbsi_beta"]
FEATURES_9_STUDY1 = ["pdbsi_beta", "beta", "pdbsi_alpha", "theta", "pdbsi_theta",
                     "pdbsi_delta", "delta", "alpha", "skewness"]
CORR_THRESHOLD = 0.6
VIF_THRESHOLD = 10.0

# ----------------------------------------------------------------------------- FM branch
FS_FM = 128                  # ST-EEGFormer input rate
FM_BANDPASS = (0.5, 63.0)    # model card: 0.5-64 Hz band-pass, per-channel z-score
FM_NOTCH = 50.0

# ----------------------------------------------------------------------------- evaluation
N_FOLDS = 5                  # subject-level StratifiedGroupKFold
N_REPEATS = 3                # -> 15 outer folds shared by every method
INNER_FOLDS = 3              # subject-level inner CV for hyperparameter search
N_ITER_SEARCH = 50           # randomized-search budget per model (as in Study 1)
SEED = 42
TASKS = ("mi", "hemi")       # mi: left- vs right-hand MI (trial label); hemi: affected side (patient label)


# ----------------------------------------------------------------------------- self-check
if __name__ == "__main__":
    def _show(label, p, required=True):
        ok = p.exists()
        flag = "OK     " if ok else ("MISSING" if required else "absent (optional)")
        print(f"{flag}  {label:<22} {p}")

    print("Folders the scripts will use:\n")
    _show("data (DATA_ROOT)", DATA_ROOT)
    _show("  sourcedata", DATA_ROOT / "sourcedata")
    n_mat = len(list((DATA_ROOT / "sourcedata").glob("sub-*/*.mat")))
    print(f"         -> {n_mat} subject .mat files found (50 released; {len(EXCLUDE_SUBJECTS)} duplicates "
          f"{list(EXCLUDE_SUBJECTS)} are skipped, so {N_SUBJECTS - len(EXCLUDE_SUBJECTS)} are used)")
    names = list(DATA_ROOT.rglob("*.edf"))[:1] + list(DATA_ROOT.rglob("*.locs"))[:1]
    print(f"         -> channel-name source: {names[0] if names else 'NONE FOUND (unzip edffile.zip or code.zip into data)'}")
    side = list(DATA_ROOT.rglob("hemiplegia.csv")) + list(DATA_ROOT.rglob("participants.tsv"))
    print(f"         -> affected-side labels: {side[0] if side else 'none: download participants.tsv from figshare into data (see README)'}")
    _show("STEEGFormer repo", STEEGFORMER_REPO)
    _show("checkpoint small", STEEGFORMER_CKPT["small"])
    _show("checkpoint base", STEEGFORMER_CKPT["base"], required=False)
    _show("checkpoint large", STEEGFORMER_CKPT["large"], required=False)
    print(f"\nOutputs go to:  {WORK_DIR}\n                {RESULTS_DIR}")

