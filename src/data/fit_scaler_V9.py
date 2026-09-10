# fit_scaler_V9.py
"""
Fits the StandardScaler for clot detection features — V9 (2-class).

V9 architectural change (from V6): drops blood entirely. The model no longer
predicts blood at inference — blood is always emitted from the DA hard reset.
This scaler is therefore fit ONLY on windows where every sample is clot (1)
or wall (2). Blood windows are excluded so the scaler's mean/std reflect the
distribution the 2-class GRU will actually see.

Output filename is distinct from V6 so both scalers can coexist:
    clot_feature_scaler_V9_2class_seq{SEQ_LEN}_{dim_str}.pkl
"""

import pandas as pd
import numpy as np
import joblib
from joblib import Parallel, delayed
from pathlib import Path
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.signal import lfilter

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"
os.environ["PYARROW_HOTFIX_DISABLED"] = "1"

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ================= CONSTANTS =================
SAMPLE_RATE = 150

# ================= IMPORT FROM gru_torch_V9 =================
# V9 uses the same feature set / extractor as V6/V9; only the classification head changes.
from src.models.gru_torch_V9 import FEATURE_SET, TOTAL_FEATURES, \
    ClotFeatureExtractor, SEQ_LEN, WINDOW_SEC, active_idx, active_dim, dim_str

# V9 training script defines STRIDE_SAMPLES; import from there to stay in sync.
from src.training.train_gru_V9 import STRIDE_SAMPLES

# ================= Derived CONFIG =================
OUTPUT_SCALER_PATH = PROJECT_ROOT / "src" / "data" / f"clot_feature_scaler_V9_2class_seq{SEQ_LEN}_{dim_str}.pkl"

print("Scaler config (V9 / 2-class):")
print(f"   FEATURE_SET     = {FEATURE_SET}")
print(f"   SEQ_LEN         = {SEQ_LEN}")
print(f"   SAMPLE_RATE     = {SAMPLE_RATE} Hz")
print(f"   WINDOW_SEC      = {WINDOW_SEC} s")
print(f"   STRIDE_SAMPLES  = {STRIDE_SAMPLES}")
print(f"   Total features  = {TOTAL_FEATURES}")
print(f"   Active features = {active_dim}  ({FEATURE_SET})")
print(f"   Label filter    = {{1: clot, 2: wall}}  (blood excluded)")

# ================= Extractor + lfilter setup =================
extractor = ClotFeatureExtractor(sample_rate=SAMPLE_RATE, window_sec=WINDOW_SEC,
                                 active_features=active_idx)
WINDOW_SAMPLES = extractor.window_size
ALPHA_FAST = extractor.alpha_fast
ALPHA_SLOW = extractor.alpha_slow

# lfilter coefficients for vectorized EMA (precomputed once)
_B_FAST = np.array([ALPHA_FAST])
_A_FAST = np.array([1.0, -(1.0 - ALPHA_FAST)])
_B_SLOW = np.array([ALPHA_SLOW])
_A_SLOW = np.array([1.0, -(1.0 - ALPHA_SLOW)])

# ================= Main Scaler Fitting =================
TRAINING_DATA_DIR = PROJECT_ROOT / "training_data"

print(f"\nLooking for parquet files in: {TRAINING_DATA_DIR}")

parquet_files = list(TRAINING_DATA_DIR.glob("*_labeled_segment.parquet"))
if not parquet_files:
    print("No *_labeled_segment.parquet files found!")
    sys.exit(1)

print(f"Found {len(parquet_files)} parquet files.")

global_features = []  # list of individual feature vectors


def _extract_one_file(file_path):
    """Extract features from a single parquet file (for parallel execution).

    V9 filters to windows where EVERY sample has label in {1, 2}. Blood samples
    are excluded from scaler fitting entirely.
    """
    try:
        df = pd.read_parquet(file_path, engine='pyarrow')
    except Exception as e:
        print(f"  Failed to read {file_path.name}: {e}")
        return []

    required_cols = ['timeInMS', 'magRLoadAdjusted', 'label']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"  {file_path.name}: Missing columns: {missing} — skipping")
        return []

    df = df[required_cols].copy().dropna().reset_index(drop=True)
    resistance = df['magRLoadAdjusted'].to_numpy(dtype=np.float32)
    labels = df['label'].to_numpy(dtype=np.int64)

    # V9: only clot(1) and wall(2). Blood(0) is excluded from the scaler.
    valid_mask = np.isin(labels, [1, 2])

    if len(resistance) < SAMPLE_RATE * WINDOW_SEC:
        return []

    # Per-worker extractor instance (avoids shared state across processes)
    ext = ClotFeatureExtractor(sample_rate=SAMPLE_RATE, window_sec=WINDOW_SEC,
                               active_features=active_idx)

    # Precompute full-run EMA arrays over the entire resistance series so the
    # EMA state at each window's end reflects the true stream history — this
    # matches how the live inference computes EMA. Blood samples are still
    # part of the EMA history; we only exclude them from the SET of windows
    # used for scaler fitting.
    r0 = float(resistance[0])
    ema_f_all, _ = lfilter(_B_FAST, _A_FAST, resistance.astype(np.float64),
                           zi=[r0 * (1.0 - ALPHA_FAST)])
    ema_s_all, _ = lfilter(_B_SLOW, _A_SLOW, resistance.astype(np.float64),
                           zi=[r0 * (1.0 - ALPHA_SLOW)])

    run_features = []

    for start in range(0, len(resistance) - WINDOW_SAMPLES + 1, STRIDE_SAMPLES):
        end = start + WINDOW_SAMPLES - 1
        # V9: require every sample in the window to be clot or wall.
        if not valid_mask[start : start + WINDOW_SAMPLES].all():
            continue

        window_res = resistance[start : start + WINDOW_SAMPLES]

        feats = ext.compute_features_from_array(
            window_res, float(ema_f_all[end]), float(ema_s_all[end]))
        if feats is not None and len(feats) == active_dim:
            run_features.append(feats)

    return run_features


# Parallel extraction across files
n_jobs = min(os.cpu_count() or 4, len(parquet_files))
print(f"\nParallelizing across {len(parquet_files)} files with {n_jobs} workers...")

results = Parallel(n_jobs=n_jobs, prefer="processes")(
    delayed(_extract_one_file)(fp) for fp in parquet_files
)

for file_features in results:
    global_features.extend(file_features)

print(f"  Total: {len(global_features)} feature vectors from {len(parquet_files)} files")

if not global_features:
    print("No features extracted! Check that training data contains clot/wall labels.")
    sys.exit(1)

X = np.array(global_features, dtype=np.float32)
print(f"\nCollected {X.shape[0]} feature vectors | shape={X.shape}")

X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Scaling
print("Fitting scaler on clot+wall feature vectors...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nAfter scaling ({X.shape[1]} active features):")
print(f"   Mean: {X_scaled.mean():.6f}   (should be very close to 0)")
print(f"   Std : {X_scaled.std():.6f}   (should be very close to 1)")

# Save the fitted scaler
joblib.dump(scaler, OUTPUT_SCALER_PATH)
print(f"\n✅ V9 (2-class) scaler fitted and saved → {OUTPUT_SCALER_PATH}")

# ================= Feature Correlation Heatmap =================
print("\nGenerating feature correlation heatmap...")
df_feat = pd.DataFrame(X_scaled, columns=[f"f{i}" for i in range(X_scaled.shape[1])])
corr_matrix = df_feat.corr().abs()

plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=0, vmax=1, square=True)
plt.title(f"V9 (2-class) Feature Correlation Heatmap ({X_scaled.shape[1]} features)")
plt.tight_layout()
plt.savefig("feature_correlation_heatmap_V9_2class.png", dpi=300, bbox_inches='tight')
plt.close()
print("   Saved → feature_correlation_heatmap_V9_2class.png")

print("\nV9 scaler fitting complete.")
