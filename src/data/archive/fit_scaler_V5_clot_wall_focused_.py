# fit_scaler_V5.py
"""
Fits the StandardScaler for clot detection features with SEQ_LEN support.
Imports FEATURE_SET and extractor from gru_torch_V5.py (single source of truth).
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os
import sys
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.signal import lfilter
from scipy import stats

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"
os.environ["PYARROW_HOTFIX_DISABLED"] = "1"

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ================= CONSTANTS =================
SAMPLE_RATE = 150
MIN_SAMPLES_FOR_FEATURE = 100

# ================= IMPORT FROM gru_torch_V5 =================
from src.models.gru_torch_V5 import FEATURE_SET, TOTAL_FEATURES, \
    ClotFeatureExtractor, SEQ_LEN, WINDOW_SEC, active_idx, active_dim, dim_str

from src.training.train_gru_V5 import STRIDE_SAMPLES

# ================= Derived CONFIG =================
OUTPUT_SCALER_PATH = PROJECT_ROOT / "src" / "data" / f"clot_feature_scaler_5s_seq{SEQ_LEN}_{dim_str}.pkl"

print("Scaler config:")
print(f"   FEATURE_SET     = {FEATURE_SET}")
print(f"   SEQ_LEN         = {SEQ_LEN}")
print(f"   SAMPLE_RATE     = {SAMPLE_RATE} Hz")
print(f"   WINDOW_SEC      = {WINDOW_SEC} s")
print(f"   STRIDE_SAMPLES  = {STRIDE_SAMPLES}")
print(f"   Total features  = {TOTAL_FEATURES}")
print(f"   Active features = {active_dim}  ({FEATURE_SET})")

# ================= Fast feature extractor (active features only) =================
_SUPPORTED_FEATURES = {0, 1, 3, 4, 5, 9, 17, 19, 20, 21, 23, 27, 28, 29, 32, 34, 38, 39, 44, 45, 52}
_unsupported = set(active_idx) - _SUPPORTED_FEATURES
if _unsupported:
    print(f"ERROR: active_idx contains features {_unsupported} not supported by fast extractor.")
    print(f"       Update compute_active_features_fast() or use full ClotFeatureExtractor.")
    sys.exit(1)

# Read EMA constants from canonical source
_extractor = ClotFeatureExtractor(sample_rate=SAMPLE_RATE, window_sec=WINDOW_SEC)
WINDOW_SAMPLES = _extractor.window_size
ALPHA_FAST = _extractor.alpha_fast
ALPHA_SLOW = _extractor.alpha_slow
del _extractor


def compute_active_features_fast(data, ema_fast, ema_slow):
    """Compute only features used by active_idx. Skips FFT, slopes, sample entropy."""
    n = len(data)
    if n < 100:
        return None

    f = np.zeros(TOTAL_FEATURES, dtype=np.float32)

    # Basic stats (f0, f1, f3, f4, f5)
    f[0] = data.mean()
    f[1] = data.std()
    f[3] = data.min()
    f[4] = data.max()
    f[5] = np.ptp(data)

    # Diff of last 500 samples (f9, f32)
    if n >= 500:
        diff_500 = np.diff(data[-500:])
        f[9] = np.mean(np.abs(diff_500))
        f[32] = np.std(diff_500)

    # Full derivative (f17, f19, f20, f21)
    deriv = np.diff(data)
    if len(deriv) > 10:
        f[17] = deriv.std()
        f[19] = np.mean(np.abs(deriv))
        f[20] = stats.skew(deriv) if len(deriv) >= 3 else 0
        f[21] = stats.kurtosis(deriv) if len(deriv) >= 4 else 0

    # EMA (f23, f27)
    f[23] = ema_slow
    f[27] = np.abs(ema_fast - ema_slow)

    # Detrended (f28, f29, f34)
    kernel = 450
    if n >= kernel:
        trend = np.convolve(data, np.ones(kernel) / kernel, 'valid')
        detr = data[-len(trend):] - trend
        r600 = detr[-min(600, len(detr)):]
        f[28] = np.std(r600)
        f[29] = np.std(r600[:300])
        f[34] = stats.skew(r600) if len(r600) >= 3 else 0

    # Percentiles (f38, f39)
    p95 = np.percentile(data, 95)
    f[38] = p95 - np.percentile(data, 5)
    f[39] = np.sum(data > p95) / n

    # Hjorth mobility & complexity (f44, f45)
    data_var = data.var() + 1e-8
    dx_var = deriv.var() + 1e-8
    ddx = np.diff(deriv)
    ddx_var = ddx.var() + 1e-8
    hjorth_mob = np.sqrt(dx_var / data_var)
    f[44] = hjorth_mob
    f[45] = np.sqrt(ddx_var / dx_var) / (hjorth_mob + 1e-8)

    # Mean abs second derivative (f52)
    if len(ddx) > 0:
        f[52] = np.mean(np.abs(ddx))

    return f[active_idx]


# ================= Main Scaler Fitting =================
TRAINING_DATA_DIR = PROJECT_ROOT / "training_data"

print(f"\nLooking for parquet files in: {TRAINING_DATA_DIR}")

parquet_files = list(TRAINING_DATA_DIR.glob("*_labeled_segment.parquet"))
if not parquet_files:
    print("No *_labeled_segment.parquet files found!")
    sys.exit(1)

print(f"Found {len(parquet_files)} parquet files.")

global_features = []  # list of individual feature vectors

for file_path in tqdm(parquet_files, desc="Processing files"):
    print(f"\nReading: {file_path.name}")
    try:
        df = pd.read_parquet(file_path, engine='pyarrow')
    except Exception as e:
        print(f"  Failed to read: {e}")
        continue

    required_cols = ['timeInMS', 'magRLoadAdjusted', 'label']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"  Missing columns: {missing} — skipping")
        continue

    df = df[required_cols].copy().dropna().reset_index(drop=True)
    resistance = df['magRLoadAdjusted'].to_numpy(dtype=np.float32)

    if len(resistance) < SAMPLE_RATE * WINDOW_SEC:
        print(f"  Too short ({len(resistance)} samples) — skipping")
        continue

    run_features = []   # single feature vectors for this run

    # Precompute lfilter coefficients for vectorized EMA
    b_f, a_f = np.array([ALPHA_FAST]), np.array([1.0, -(1.0 - ALPHA_FAST)])
    b_s, a_s = np.array([ALPHA_SLOW]), np.array([1.0, -(1.0 - ALPHA_SLOW)])

    for start in range(0, len(resistance) - WINDOW_SAMPLES + 1, STRIDE_SAMPLES):
        window_res = resistance[start : start + WINDOW_SAMPLES]

        # Vectorized EMA via lfilter
        r0 = float(window_res[0])
        ema_f, _ = lfilter(b_f, a_f, window_res.astype(np.float64), zi=[r0 * (1.0 - ALPHA_FAST)])
        ema_s, _ = lfilter(b_s, a_s, window_res.astype(np.float64), zi=[r0 * (1.0 - ALPHA_SLOW)])

        feats = compute_active_features_fast(window_res, float(ema_f[-1]), float(ema_s[-1]))
        if feats is not None:
            run_features.append(feats)

    global_features.extend(run_features)
    print(f"  → {len(run_features)} feature vectors extracted")

if not global_features:
    print("No features extracted!")
    sys.exit(1)

X = np.array(global_features, dtype=np.float32)   # (N, active_dim)
print(f"\nCollected {X.shape[0]} feature vectors | shape={X.shape}")

X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Scaling (fit on individual vectors — same stats as flattened sequences)
print("Fitting scaler on feature vectors...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nAfter scaling ({X.shape[1]} active features):")
print(f"   Mean: {X_scaled.mean():.6f}   (should be very close to 0)")
print(f"   Std : {X_scaled.std():.6f}   (should be very close to 1)")

# Save the fitted scaler
joblib.dump(scaler, OUTPUT_SCALER_PATH)
print(f"\n✅ Scaler fitted and saved → {OUTPUT_SCALER_PATH}")

# ================= Feature Correlation Heatmap =================
print("\nGenerating feature correlation heatmap...")
df_feat = pd.DataFrame(X_scaled, columns=[f"f{i}" for i in range(X_scaled.shape[1])])
corr_matrix = df_feat.corr().abs()

plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=0, vmax=1, square=True)
plt.title(f"Feature Correlation Heatmap ({X_scaled.shape[1]} features)")
plt.tight_layout()
plt.savefig("feature_correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()
print("   Saved → feature_correlation_heatmap.png")

print("\nScaler fitting complete.")