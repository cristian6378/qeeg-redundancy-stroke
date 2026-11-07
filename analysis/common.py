"""Shared settings of the redundancy study (Studies A-C).

The study reuses the signal processing and the 17 feature definitions of ../pipeline (config.py,
data.py, features.py) so that every feature is computed exactly as in the companion analyses, but
it runs different experiments: (A) a comparison of feature-selection strategies on the synthetic
benchmark, (B) measurement properties of the features on real acute-stroke EEG (reliability, epoch
length, electrode sparsity, channel dropout), and (C) association with clinical severity.

Paths (override with environment variables):
  SYNTH_FEATURES  synthetic feature table electra_features_realistic.csv (thesis repository)
  MDPI_WORK       intermediate files (default <repo>/work)
  MDPI_RESULTS    results of a new run (default <repo>/output; the published results are in <repo>/results)
  the real EEG location is the one of ../pipeline/config.py (STROKE_EEG_ROOT)
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
_NEEDED = ("config.py", "data.py", "features.py", "selection.py", "stats_tools.py")


def _find_pipeline():
    """Folder with the main pipeline modules: PIPELINE_DIR if set, else <repo>/pipeline, else the folder
    that contains this folder, else a sibling folder."""
    env = os.environ.get("PIPELINE_DIR", "").strip().strip('"')
    cands = ([Path(env)] if env else []) + [ROOT / "pipeline", ROOT]
    cands += sorted(p for p in ROOT.iterdir() if p.is_dir() and p != HERE)
    for c in cands:
        if all((c / f).exists() for f in _NEEDED):
            return c.resolve()
    raise SystemExit("Cannot find the pipeline code (config.py, data.py, features.py, selection.py, "
                     f"stats_tools.py) next to {HERE}.\nKeep the analysis and pipeline folders side by side, "
                     "or set PIPELINE_DIR to the folder that contains config.py.")


PIPELINE_DIR = _find_pipeline()
sys.path.insert(0, str(PIPELINE_DIR))

import config as C   # noqa: E402  (pipeline configuration)

WORK = Path(os.environ.get("MDPI_WORK") or ROOT / "work").expanduser().resolve()
OUT = Path(os.environ.get("MDPI_RESULTS") or ROOT / "output").expanduser().resolve()
SYNTH_CSV = Path(os.environ.get("SYNTH_FEATURES") or ROOT / "data" / "synthetic" / "electra_features_realistic.csv")
SEED = 42

F17 = list(C.FEATURES_17)
FAMILIES = {
    "spectral": ["delta", "theta", "alpha", "beta"],
    "ratio": ["delta_alpha", "theta_alpha", "dt_ab"],
    "distribution": ["kurtosis", "skewness"],
    "complexity": ["sample_entropy", "higuchi_fd"],
    "connectivity": ["weighted_phase_lag_index", "magnitude_squared_coherence"],
    "asymmetry": ["pdbsi_delta", "pdbsi_theta", "pdbsi_alpha", "pdbsi_beta"],
}
FAMILY_OF = {f: g for g, fs in FAMILIES.items() for f in fs}
# a-priori clinical panel (fixed before any analysis): slowing ratios used in ischaemia/LVO work,
# alpha loss, and band-wise hemispheric asymmetry (ELECTRA-STROKE; Finnigan & van Putten 2013)
CLINICAL_PANEL = ["delta_alpha", "dt_ab", "alpha", "pdbsi_delta", "pdbsi_theta", "pdbsi_alpha"]
# strictly positive, right-skewed features: analysed on the log scale
LOG_FEATURES = ["delta", "theta", "alpha", "beta", "delta_alpha", "theta_alpha", "dt_ab", "kurtosis"]
LABELS = {"delta": r"$\delta$", "theta": r"$\theta$", "alpha": r"$\alpha$", "beta": r"$\beta$",
          "delta_alpha": "DAR", "theta_alpha": "TAR", "dt_ab": "DTABR", "kurtosis": "Kurtosis",
          "skewness": "Skewness", "sample_entropy": "SampEn", "higuchi_fd": "HFD",
          "weighted_phase_lag_index": "wPLI", "magnitude_squared_coherence": "MSC",
          "pdbsi_delta": r"pdBSI-$\delta$", "pdbsi_theta": r"pdBSI-$\theta$",
          "pdbsi_alpha": r"pdBSI-$\alpha$", "pdbsi_beta": r"pdBSI-$\beta$"}

# wearable montages (names as in the dataset after upper-casing); all are subsets of the 29 channels
MONTAGES = {
    "M29": None,                                                     # all retained channels
    "M19": ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T3", "C3", "CZ", "C4", "T4", "T5", "P3", "PZ",
            "P4", "T6", "O1", "O2"],                                 # classic 10-20 (CPz reference absent)
    "M8": ["FC3", "FCZ", "FC4", "C3", "CZ", "C4", "CP3", "CP4"],     # sensorimotor headband
    "M4F": ["FP1", "FP2", "F7", "F8"],                               # frontal 4 (Schleiger et al. 2014)
    "M4C": ["C3", "C4", "P3", "P4"],                                 # centro-parietal 4
    "M2": ["C3", "C4"],                                              # one homologous pair
}
EPOCH_LENGTHS_S = (1, 2, 4, 8)
DROPOUT_LEVELS = (2, 4, 8)          # channels removed at random from the 29
DROPOUT_REPS = 3
DROPOUT_TRIALS = 20                 # first 20 trials of each patient

# feature-extraction cost per 4-s, 29-channel window at 500 Hz (median ms, one CPU thread), measured by
# pipeline/06_efficiency.py; replaced by the measured file if it is available
COST_MS = {"band powers (+ratios)": 1.07, "kurtosis": 2.48, "skewness": 2.81, "sample entropy": 351.79,
           "Higuchi FD": 1.54, "wPLI": 32.49, "MSC": 107.87, "pDBSI (4 bands)": 1.36}
COST_GROUP = {**{f: "band powers (+ratios)" for f in FAMILIES["spectral"] + FAMILIES["ratio"]},
              "kurtosis": "kurtosis", "skewness": "skewness", "sample_entropy": "sample entropy",
              "higuchi_fd": "Higuchi FD", "weighted_phase_lag_index": "wPLI",
              "magnitude_squared_coherence": "MSC", **{f: "pDBSI (4 bands)" for f in FAMILIES["asymmetry"]}}


def cost_table():
    p = C.RESULTS_DIR / "efficiency_feature_components.csv"
    if p.exists():
        import pandas as pd
        d = pd.read_csv(p)
        return dict(zip(d.component, d.median_ms))
    return dict(COST_MS)


def subset_cost(features, table=None):
    """ms per window to compute a feature subset; pDBSI needs the band powers too."""
    table = table or cost_table()
    groups = {COST_GROUP[f] for f in features}
    if "pDBSI (4 bands)" in groups:
        groups.add("band powers (+ratios)")
    return float(sum(table[g] for g in groups))


def ensure_dirs():
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
