# fit_scaler_V4.py
# FIT THE NORMALIZER — Imports REDUCE_DIM and extractor from gru_torch_V4.py
# All magic numbers replaced with named constants.

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os
import sys
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"
os.environ["PYARROW_HOTFIX_DISABLED"] = "1"

# Add project root to Python path so "src" can be found
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # goes up from src/data/ → PyTorch_3
sys.path.insert(0, str(PROJECT_ROOT))

# ================= CONSTANTS =================
SAMPLE_RATE = 150
WINDOW_SEC = 5.0
STRIDE_SAMPLES = 30
FEATURE_DIM = 40                    # Total features before reduction
MIN_SAMPLES_FOR_FEATURE = 100

# ================= IMPORT FROM gru_torch_V4.py =================
from src.models.gru_torch_V5 import REDUCE_DIM, ClotFeatureExtractor

# ================= Derived CONFIG =================
# Compute active dimension dynamically from the extractor (single source of truth)
extractor_temp = ClotFeatureExtractor(sample_rate=SAMPLE_RATE, window_sec=WINDOW_SEC)
zero_idx = extractor_temp.zero_idx
active_dim = FEATURE_DIM - len(zero_idx) if REDUCE_DIM else FEATURE_DIM
dim_str = f"red{active_dim}" if REDUCE_DIM else f"{FEATURE_DIM}"

# OUTPUT_SCALER_PATH = f"clot_feature_scaler_{int(WINDOW_SEC)}s_{dim_str}.pkl"
OUTPUT_SCALER_PATH = PROJECT_ROOT / "src" / "data" / f"clot_feature_scaler_5s_{dim_str}.pkl"

print(f"Scaler config:")
print(f"   REDUCE_DIM      = {REDUCE_DIM}")
print(f"   SAMPLE_RATE     = {SAMPLE_RATE} Hz")
print(f"   WINDOW_SEC      = {WINDOW_SEC} s")
print(f"   STRIDE_SAMPLES  = {STRIDE_SAMPLES}")
print(f"   Total features  = {FEATURE_DIM}")
print(f"   Active features = {active_dim}")

# ================= Main Scaler Fitting =================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DATA_DIR = PROJECT_ROOT / "training_data"

print(f"\nLooking for parquet files in: {TRAINING_DATA_DIR}")

parquet_files = list(TRAINING_DATA_DIR.glob("*_labeled_segment.parquet"))
if not parquet_files:
    print("No *_labeled_segment.parquet files found!")
    exit(1)

print(f"Found {len(parquet_files)} parquet files.")

global_features = []

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

    extractor = ClotFeatureExtractor(sample_rate=SAMPLE_RATE, window_sec=WINDOW_SEC)
    run_features = []

    window_samples = extractor.window_size
    for start in range(0, len(resistance) - window_samples + 1, STRIDE_SAMPLES):
        window_res = resistance[start : start + window_samples]
        extractor.reset()

        for r in window_res:
            extractor.update(r)

        feats = extractor.compute_features()   # always returns FEATURE_DIM (40)

        if feats is not None and len(feats) == FEATURE_DIM:
            run_features.append(feats)

    print(f"  → {len(run_features)} feature vectors extracted")
    global_features.extend(run_features)

if not global_features:
    print("No features extracted!")
    exit(1)

X_full = np.array(global_features, dtype=np.float32)
print(f"\nCollected {X_full.shape[0]} windows × {FEATURE_DIM} features")

X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)

# === DROP UNUSED FEATURES BEFORE SCALING ===
if REDUCE_DIM:
    active_idx = [i for i in range(FEATURE_DIM) if i not in zero_idx]
    X = X_full[:, active_idx]
    print(f"Dropped {len(zero_idx)} unused features → scaling on {X.shape[1]} active features")
else:
    X = X_full
    print(f"Keeping full {FEATURE_DIM} features for scaling")

# Fit scaler ONLY on the active features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.fit_transform(X)          # ← Important: fit_transform

# Save scaler
print(f"\nAfter scaling ({X_scaled.shape[1]} features):")
print(f"   Mean: {X_scaled.mean():.6f}   (should be very close to 0)")
print(f"   Std : {X_scaled.std():.6f}   (should be very close to 1)")

# Save the fitted scaler (it now contains the correct mean_ and scale_)
joblib.dump(scaler, OUTPUT_SCALER_PATH)
print(f"\n✅ Scaler fitted and saved → {OUTPUT_SCALER_PATH}")

# ================= Feature Correlation Heatmap =================
print("\nGenerating feature correlation heatmap...")
df_feat = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
corr_matrix = df_feat.corr().abs()

plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=0, vmax=1, square=True)
plt.title(f"Feature Correlation Heatmap ({X.shape[1]} features)")
plt.tight_layout()
plt.savefig("feature_correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()
print("   Saved → feature_correlation_heatmap.png")

print("\nScaler fitting complete.")