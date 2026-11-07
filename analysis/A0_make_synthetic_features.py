"""Study A, step 0: rebuild the synthetic feature table electra_features_realistic.csv.

Verbatim copy of the generation cell of the thesis notebook synthetic_df_feature_extraction.ipynb
(cell 48), with only the input and output paths made configurable. It reads the four class files of
the Kaggle synthetic EEG stroke dataset (Wood, 2025; 500 trials x 16 channels x 1024 samples at
256 Hz per class), adds white Gaussian noise with SD = 0.05 x the global signal SD (seed 42), applies
the 0.1-Hz high-pass, 50-Hz notch and 40-Hz low-pass filters, and extracts the 17 features.

  python A0_make_synthetic_features.py                    (downloads the dataset with kagglehub)
  python A0_make_synthetic_features.py --kaggle-dir PATH  (folder that contains the four class files)
Output: <repo>/data/synthetic/electra_features_realistic.csv  (the input of A1_selection_synthetic.py)
"""
import argparse
from pathlib import Path

import common as K

CLASS_FILES = ("hemorrhagic.csv", "ischemic.csv", "normal.csv", "tia.csv")


def find_class_folder(root):
    """Folder under root that contains the four class files; the thesis used 'new_dataset'."""
    root = Path(root)
    hits = sorted({p.parent for p in root.rglob("hemorrhagic.csv") if all((p.parent / f).exists() for f in CLASS_FILES)})
    if not hits:
        raise SystemExit(f"no folder with {', '.join(CLASS_FILES)} under {root}")
    if len(hits) > 1:
        preferred = [h for h in hits if h.name == "new_dataset"]
        if not preferred:
            raise SystemExit("several candidate folders found; choose one with --kaggle-dir:\n  " +
                             "\n  ".join(map(str, hits)))
        hits = preferred
    return hits[0]


_ap = argparse.ArgumentParser()
_ap.add_argument("--kaggle-dir", default=None, help="folder that contains (somewhere below it) the four class files")
_ap.add_argument("--out", default=str(K.SYNTH_CSV))
_args = _ap.parse_args()
if _args.kaggle_dir is None:
    import kagglehub   # pip install kagglehub
    _root = kagglehub.dataset_download("aniellewood/eeg-stroke-dataset")
    print("Kaggle dataset downloaded to:", _root)
else:
    _root = _args.kaggle_dir
KAGGLE_DIR = find_class_folder(_root)
print("using class files in:", KAGGLE_DIR)
OUT_FILE = Path(_args.out)

# ----------------------------------------------------------------------------- thesis code (unchanged below)
"""
Complete EEG Processing Pipeline with Realistic Data Generation
================================================================
This script provides both:
1. Clean feature extraction from synthetic EEG signals
2. Optional realistic data generation with noise, overlap, and imbalance
"""

import pandas as pd
import numpy as np
from scipy import signal as sig
from scipy.signal import welch, csd, hilbert
from scipy.stats import kurtosis, skew
from scipy.fft import rfft, rfftfreq  # noqa: F401  (kept verbatim from the thesis)
from antropy import sample_entropy as ant_sample_entropy
from itertools import product
try:
    from tqdm import tqdm
except ImportError:  # progress bars are optional
    def tqdm(x, **k):
        return x
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================
MAKE_REALISTIC = True  # Set to False for clean synthetic data
NOISE_LEVEL = 0.05  # White noise ratio
OVERLAP_STRENGTH = 0.35  # Class overlap amount (0-1)
OUTLIER_RATIO = 0.05  # Proportion of artifact-corrupted samples
MISSING_RATIO = 0.02  # Proportion of missing values
LABEL_NOISE_RATIO = 0.05  # Diagnostic uncertainty
CREATE_IMBALANCE = True  # Create realistic class distribution
SEED = 42  # Random seed for reproducibility

# Sampling parameters
FS = 256  # Sampling frequency
N_CHANNELS = 16
N_SAMPLES = 1024

# Filter parameters
HP_CUTOFF = 0.1
LP_CUTOFF = 40.0
NOTCH_FREQ = 50.0
FILTER_ORDER = 4

# Band definitions
BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 18),
}

# Homologous channel pairs for pDBSI
HOMOLOGOUS_PAIRS = [
    (0, 1),    # Fp1 - Fp2
    (2, 3),    # F3 - F4
    (4, 5),    # C3 - C4
    (6, 7),    # P3 - P4
    (8, 9),    # O1 - O2
    (10, 11),  # T3 - T4
    (12, 13)   # T5 - T6
]

print("="*70)
print("EEG PROCESSING PIPELINE WITH REALISTIC DATA GENERATION")
print("="*70)
print(f"\nMode: {'REALISTIC (Dirty Data)' if MAKE_REALISTIC else 'CLEAN (Synthetic)'}")
if MAKE_REALISTIC:
    print(f"  • Noise level: {NOISE_LEVEL*100:.1f}%")
    print(f"  • Class overlap: {OVERLAP_STRENGTH*100:.1f}%")
    print(f"  • Outliers: {OUTLIER_RATIO*100:.1f}%")
    print(f"  • Missing data: {MISSING_RATIO*100:.1f}%")
    print(f"  • Label noise: {LABEL_NOISE_RATIO*100:.1f}%")

np.random.seed(SEED)
# ============================================
# 1. DATA LOADING
# ============================================
print("\n" + "="*70)
print("STEP 1: LOADING DATA")
print("="*70)

df_hemorrhagic = pd.read_csv(KAGGLE_DIR / 'hemorrhagic.csv')
df_ischemic = pd.read_csv(KAGGLE_DIR / 'ischemic.csv')
df_normal = pd.read_csv(KAGGLE_DIR / 'normal.csv')
df_tia = pd.read_csv(KAGGLE_DIR / 'tia.csv')

def convert_df_to_np_array(df):
    N_METADATA_COLS = 3
    eeg_df = df.iloc[:, :-N_METADATA_COLS]
    eeg_flat = eeg_df.values
    N_trials = eeg_flat.shape[0]
    
    label_df = df['label']
    label_flat = label_df.values
    
    eeg_array = eeg_flat.reshape(N_trials, N_CHANNELS, N_SAMPLES)
    label_array = label_flat.reshape(N_trials, 1)
    return eeg_array, label_array

eeg_array_ischemic, label_array_ischemic = convert_df_to_np_array(df_ischemic)
eeg_array_hemorrhagic, label_array_hemorrhagic = convert_df_to_np_array(df_hemorrhagic)
eeg_array_normal, label_array_normal = convert_df_to_np_array(df_normal)
eeg_array_tia, label_array_tia = convert_df_to_np_array(df_tia)

X_final = np.concatenate((eeg_array_ischemic, eeg_array_hemorrhagic, 
                          eeg_array_normal, eeg_array_tia), axis=0)
Y_final = np.concatenate((label_array_ischemic, label_array_hemorrhagic, 
                          label_array_normal, label_array_tia), axis=0)

print(f"✓ Loaded {X_final.shape[0]} trials")
print(f"  Shape: {X_final.shape} (Trials, Channels, Samples)")
print(f"  Classes: {np.unique(Y_final)}")

# ============================================
# 2. NOISE INJECTION
# ============================================
print("\n" + "="*70)
print("STEP 2: NOISE INJECTION")
print("="*70)

def add_flexible_white_noise(signal, noise_ratio=0.05, seed=42):
    signal_float = signal.astype(np.float64)
    signal_mean = np.mean(signal_float, axis=2, keepdims=True)
    signal_centered = signal_float - signal_mean
    
    signal_std_centered = np.std(signal_centered)
    noise_sigma = signal_std_centered * noise_ratio
    
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=noise_sigma, size=signal.shape)
    
    return signal_float + noise

X_noisy = add_flexible_white_noise(X_final, noise_ratio=NOISE_LEVEL, seed=42)
print(f"✓ Added white noise (ratio: {NOISE_LEVEL})")

# ============================================
# 3. FILTERING
# ============================================
print("\n" + "="*70)
print("STEP 3: FILTERING")
print("="*70)

def filter_highpass(data, fs, cutoff_freq=0.1, order=4, axis=2):
    nyquist = fs / 2
    normalized_cutoff = cutoff_freq / nyquist
    b, a = sig.butter(order, normalized_cutoff, 'highpass', analog=False)
    return sig.filtfilt(b, a, data, axis=axis)

def filter_notch(data, fs, notch_freq=50.0, Q=30.0, axis=2):
    b, a = sig.iirnotch(notch_freq, Q, fs)
    return sig.filtfilt(b, a, data, axis=axis)

def filter_lowpass(data, fs, cutoff_freq=40.0, order=4, axis=2):
    nyquist = fs / 2
    normalized_cutoff = cutoff_freq / nyquist
    b, a = sig.butter(order, normalized_cutoff, 'lowpass', analog=False)
    return sig.filtfilt(b, a, data, axis=axis)

print(f"Applying High-Pass filter ({HP_CUTOFF} Hz)...")
X_hp = filter_highpass(X_noisy, FS, HP_CUTOFF, FILTER_ORDER)

print(f"Applying Notch filter ({NOTCH_FREQ} Hz)...")
X_notch = filter_notch(X_hp, FS, NOTCH_FREQ)

print(f"Applying Low-Pass filter ({LP_CUTOFF} Hz)...")
X_cleaned = filter_lowpass(X_notch, FS, LP_CUTOFF, FILTER_ORDER)

print("✓ Filtering complete")

# ============================================
# 4. FEATURE EXTRACTION
# ============================================
print("\n" + "="*70)
print("STEP 4: FEATURE EXTRACTION")
print("="*70)

feature_names = []

# --- 4.1 Band Powers ---
print("Extracting band powers...")
def extract_bandpowers(X, fs):
    T, C, N = X.shape
    freqs = rfftfreq(N, d=1/fs)
    
    band_idx = {name: (freqs >= fmin) & (freqs < fmax) 
                for name, (fmin, fmax) in BANDS.items()}
    bp = {band: np.zeros((T, C)) for band in BANDS}
    
    for t in range(T):
        for c in range(C):
            sig = X[t, c]
            fft_vals = np.abs(np.fft.rfft(sig))**2
            
            for name, idx in band_idx.items():
                bp[name][t, c] = np.sum(fft_vals[idx])
    
    return bp

def extract_band_ratios(bp):
    ratios = {}
    ratios["delta_alpha"] = bp["delta"] / (bp["alpha"] + 1e-12)
    ratios["theta_alpha"] = bp["theta"] / (bp["alpha"] + 1e-12)
    ratios["dt_ab"] = (bp["delta"] + bp["theta"]) / (bp["alpha"] + bp["beta"] + 1e-12)
    return ratios

def aggregate_median(feat):
    return np.median(feat, axis=1)

bp = extract_bandpowers(X_cleaned, FS)
ratios = extract_band_ratios(bp)

features = {}
features["delta"] = aggregate_median(bp["delta"])
features["theta"] = aggregate_median(bp["theta"])
features["alpha"] = aggregate_median(bp["alpha"])
features["beta"] = aggregate_median(bp["beta"])

for k, v in ratios.items():
    features[k] = aggregate_median(v)

feature_names = list(features.keys())
X_feat = np.vstack([features[k] for k in feature_names]).T

print(f"✓ Extracted {len(BANDS)} band powers and {len(ratios)} ratios")

# --- 4.2 Statistical Features ---
print("Extracting statistical features...")

kurtosis_feat = aggregate_median(kurtosis(X_cleaned, axis=2, fisher=False))
skewness_feat = aggregate_median(skew(X_cleaned, axis=2, bias=False))

feature_names.extend(["kurtosis", "skewness"])
X_feat = np.column_stack([X_feat, kurtosis_feat, skewness_feat])

print("✓ Extracted kurtosis and skewness")

# --- 4.3 Sample Entropy ---
print("Extracting sample entropy...")

def extract_sample_entropy(X_cpu):
    T, C, _ = X_cpu.shape
    out = np.zeros((T, C), dtype=np.float32)
    all_indices = list(product(range(T), range(C)))
    
    for t, c in tqdm(all_indices, desc="Sample Entropy", unit="segment"):
        sig = X_cpu[t, c].astype(np.float32)
        tolerance_val = (0.2 * np.std(sig)).item()
        out[t, c] = ant_sample_entropy(sig, order=2, tolerance=tolerance_val)
    
    return out

sample_entropy_feat = aggregate_median(extract_sample_entropy(X_cleaned))
feature_names.append("sample_entropy")
X_feat = np.column_stack([X_feat, sample_entropy_feat])

print("✓ Extracted sample entropy")


# --- 4.4 WPLI ---
print("Extracting WPLI...")

def wpli(signal1, signal2):
    a1 = hilbert(signal1)
    a2 = hilbert(signal2)
    im = np.imag(a1 * np.conj(a2))
    num = np.abs(np.mean(im))
    denom = np.mean(np.abs(im)) + 1e-12
    return num / denom

def extract_wpli(X):
    T, C, S = X.shape
    out = np.zeros(T)
    
    for t in range(T):
        pairs = []
        for i in range(C):
            for j in range(i+1, C):
                pairs.append(wpli(X[t,i], X[t,j]))
        out[t] = np.median(pairs)
    
    return out

wpli_feat = extract_wpli(X_cleaned)
feature_names.append("weighted_phase_lag_index")
X_feat = np.column_stack([X_feat, wpli_feat])

print("✓ Extracted WPLI")

# --- 4.5 Coherence ---
print("Extracting magnitude-squared coherence...")

def magnitude_squared_coherence(sig1, sig2, fs, nperseg=None):
    f, Pxx = welch(sig1, fs=fs, nperseg=nperseg)
    f, Pyy = welch(sig2, fs=fs, nperseg=nperseg)
    f, Pxy = csd(sig1, sig2, fs=fs, nperseg=nperseg)
    
    Cxy = np.abs(Pxy)**2 / (Pxx * Pyy + 1e-12)
    return np.mean(Cxy)

def extract_coherence(X, fs, nperseg=None):
    T, C, S = X.shape
    out = np.zeros(T)
    
    for t in range(T):
        pairs = []
        for i in range(C):
            for j in range(i+1, C):
                pairs.append(magnitude_squared_coherence(X[t,i], X[t,j], fs, nperseg))
        out[t] = np.median(pairs)
    
    return out

msc_feat = extract_coherence(X_cleaned, FS)
feature_names.append("magnitude_squared_coherence")
X_feat = np.column_stack([X_feat, msc_feat])

print("✓ Extracted coherence")

# --- 4.6 Higuchi Fractal Dimension (Optimized) ---
print("Extracting Higuchi Fractal Dimension...")

def hfd(X, k_max=10):
    """
    Optimized Higuchi Fractal Dimension using NumPy vectorization.
    X: 1D array (samples)
    k_max: Max scale (usually 6-10 for EEG)
    """
    L = []
    x = np.array(X)
    N = x.size
    
    for k in range(1, k_max + 1):
        Lk = []
        for m in range(k):
            # Optimized slicing to calculate differences
            # Equivalent to: sum(|x(m+ik) - x(m+(i-1)k)|)
            n_max = int(np.floor((N - m - 1) / k))
            if n_max > 0:
                # This slice gets all points at scale k
                idx = m + np.arange(n_max + 1) * k
                diffs = np.abs(np.diff(x[idx]))
                
                # Normalization factor for Higuchi's formula
                norm = (N - 1) / (n_max * k)
                Lk.append(np.sum(diffs) * norm / k)
        
        if Lk:
            L.append(np.mean(Lk))

    # HFD is the slope of the log-log plot of L and 1/k
    if len(L) > 1:
        coeffs = np.polyfit(np.log(1 / np.arange(1, len(L) + 1)), np.log(L), 1)
        return coeffs[0]
    return 0

def extract_hfd(X, k_max=10):
    """
    Wrapper for (T, C, S) data structure.
    Returns median HFD across all channels per trial.
    """
    T, C, S = X.shape
    out = np.zeros(T)
    
    for t in range(T):
        # Calculate HFD for each channel in this trial
        channel_hfds = [hfd(X[t, i], k_max) for i in range(C)]
        out[t] = np.median(channel_hfds)
    
    return out

# Execution
hfd_feat = extract_hfd(X_cleaned, k_max=10)
feature_names.append("higuchi_fractal_dimension")
X_feat = np.column_stack([X_feat, hfd_feat])

print("✓ Extracted HFD")

# --- 4.7 pDBSI ---
print("Extracting pDBSI...")

def calculate_pdbsi_features(bp, homologous_pairs=HOMOLOGOUS_PAIRS, aggregate_func=np.median):
    features = {}
    for band_name, band_power_matrix in bp.items():
        T, C = band_power_matrix.shape
        N_pairs = len(homologous_pairs)
        pDBSI_band = np.zeros((T, N_pairs))
        
        for i, (L_idx, R_idx) in enumerate(homologous_pairs):
            P_L = band_power_matrix[:, L_idx]
            P_R = band_power_matrix[:, R_idx]
            numerator = np.abs(P_L - P_R)
            denominator = P_L + P_R + 1e-12
            pDBSI_band[:, i] = numerator / denominator
        
        final_feature = aggregate_func(pDBSI_band, axis=1)
        feature_name = f"pdbsi_{band_name}"
        features[feature_name] = final_feature
    
    feature_names_pdbsi = list(features.keys())
    X_feat_pdbsi = np.vstack([features[k] for k in feature_names_pdbsi]).T
    
    return X_feat_pdbsi, feature_names_pdbsi

X_feat_pdbsi, pdbsi_feature_names = calculate_pdbsi_features(bp)
feature_names.extend(pdbsi_feature_names)
X_feat = np.column_stack([X_feat, X_feat_pdbsi])

print("✓ Extracted pDBSI features")

print(f"\n✓ Total features extracted: {len(feature_names)}")

# ============================================
# 5. CREATE DATAFRAME
# ============================================
print("\n" + "="*70)
print("STEP 5: CREATING DATAFRAME")
print("="*70)

df = pd.DataFrame(X_feat, columns=feature_names)
df['label'] = Y_final.flatten()
# Create a clean copy before making realistic modifications for visualizations
df_clean = df.copy()

print(f"✓ Created dataframe with shape: {df.shape}")
print(f"  Features: {len(feature_names)}")
print(f"  Samples: {len(df)}")

# ============================================
# 7. SAVE
# ============================================
print("\n" + "="*70)
print("STEP 7: SAVING")
print("="*70)

output_filename = OUT_FILE
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_filename, index=False)

print(f"✓ Saved to '{output_filename}'")
# print(f"\nFinal dataset:")
# print(f"  Shape: {df.shape}")
# print(f"  Features: {len(feature_names)}")
# print(f"  Missing values: {df.isnull().sum().sum()}")
# print(f"\n  Class distribution:")
# for label, count in df['label'].value_counts().sort_index().items():
#     print(f"    {label:12s}: {count:4d} ({count/len(df)*100:5.1f}%)")

print("\n" + "="*70)
print("✓ PIPELINE COMPLETE")
print("="*70)