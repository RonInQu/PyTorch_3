# fit_scaler_V6.py
"""
Fits the StandardScaler for clot detection features — V6.
Uses ClotFeatureExtractor.compute_features_from_array() for efficient batch extraction.
Vectorized EMA via scipy.signal.lfilter (no per-sample Python loop).
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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"
os.environ["PYARROW_HOTFIX_DISABLED"] = "1"

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ================= CONSTANTS =================
SAMPLE_RATE = 150

# ================= IMPORT FROM gru_torch_V6 =================
from src.models.gru_torch_V6 import FEATURE_SET, TOTAL_FEATURES, \
    ClotFeatureExtractor, SEQ_LEN, WINDOW_SEC, active_idx, active_dim, dim_str

from src.training.train_gru_V6 import STRIDE_SAMPLES

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
TRAINING_DATA_DIR = PROJECT_ROOT / "training_data_denoised"

print(f"\nLooking for parquet files in: {TRAINING_DATA_DIR}")

parquet_files = list(TRAINING_DATA_DIR.glob("*_labeled_segment_denoised.parquet"))
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

    run_features = []

    for start in range(0, len(resistance) - WINDOW_SAMPLES + 1, STRIDE_SAMPLES):
        window_res = resistance[start : start + WINDOW_SAMPLES]

        # Vectorized EMA via lfilter (per-window reset)
        r0 = float(window_res[0])
        ema_f, _ = lfilter(_B_FAST, _A_FAST, window_res.astype(np.float64),
                           zi=[r0 * (1.0 - ALPHA_FAST)])
        ema_s, _ = lfilter(_B_SLOW, _A_SLOW, window_res.astype(np.float64),
                           zi=[r0 * (1.0 - ALPHA_SLOW)])

        feats = extractor.compute_features_from_array(
            window_res, float(ema_f[-1]), float(ema_s[-1]))
        if feats is not None and len(feats) == active_dim:
            run_features.append(feats)

    global_features.extend(run_features)
    print(f"  → {len(run_features)} feature vectors extracted")

if not global_features:
    print("No features extracted!")
    sys.exit(1)

X = np.array(global_features, dtype=np.float32)
print(f"\nCollected {X.shape[0]} feature vectors | shape={X.shape}")

X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Scaling
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
