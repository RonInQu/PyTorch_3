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
from src.models.gru_torch_V5 import FEATURE_SET, FEATURE_SETS, TOTAL_FEATURES, \
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

# ================= Main Scaler Fitting =================
TRAINING_DATA_DIR = PROJECT_ROOT / "training_data"

print(f"\nLooking for parquet files in: {TRAINING_DATA_DIR}")

parquet_files = list(TRAINING_DATA_DIR.glob("*_labeled_segment.parquet"))
if not parquet_files:
    print("No *_labeled_segment.parquet files found!")
    sys.exit(1)

print(f"Found {len(parquet_files)} parquet files.")

global_features = []  # list of sequences

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
    run_features = []   # single feature vectors for this run

    window_samples = extractor.window_size
    for start in range(0, len(resistance) - window_samples + 1, STRIDE_SAMPLES):
        window_res = resistance[start : start + window_samples]
        extractor.reset()

        for r in window_res:
            extractor.update(r)

        feats_all = extractor.compute_features()

        if feats_all is not None and len(feats_all) == TOTAL_FEATURES:
            feats = feats_all[active_idx]
            run_features.append(feats)

    # Build sequences from run_features
    for i in range(SEQ_LEN - 1, len(run_features)):
        seq = np.array(run_features[i - SEQ_LEN + 1 : i + 1])   # (SEQ_LEN, active_dim)
        global_features.append(seq)

    print(f"  → {len(run_features)} single vectors → {max(0, len(run_features) - SEQ_LEN + 1)} sequences")

if not global_features:
    print("No features extracted!")
    sys.exit(1)

X_seq = np.array(global_features, dtype=np.float32)   # (N, SEQ_LEN, active_dim)
print(f"\nCollected {X_seq.shape[0]} sequences | shape={X_seq.shape}")

X_seq = np.nan_to_num(X_seq, nan=0.0, posinf=0.0, neginf=0.0)

# Scaling (flatten → scale → reshape)
print("Loading scaler and scaling sequences...")
N, S, F = X_seq.shape
X_flat = X_seq.reshape(-1, F)

scaler = StandardScaler()
X_scaled_flat = scaler.fit_transform(X_flat)
X_scaled = X_scaled_flat.reshape(N, S, F)

print(f"\nAfter scaling ({F} active features):")
print(f"   Mean: {X_scaled.mean():.6f}   (should be very close to 0)")
print(f"   Std : {X_scaled.std():.6f}   (should be very close to 1)")

# Save the fitted scaler
joblib.dump(scaler, OUTPUT_SCALER_PATH)
print(f"\n✅ Scaler fitted and saved → {OUTPUT_SCALER_PATH}")

# ================= Feature Correlation Heatmap (on last timestep) =================
print("\nGenerating feature correlation heatmap (last timestep of each sequence)...")
last_timestep = X_scaled[:, -1, :]   # (N, active_dim)
df_feat = pd.DataFrame(last_timestep, columns=[f"f{i}" for i in range(last_timestep.shape[1])])
corr_matrix = df_feat.corr().abs()

plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=0, vmax=1, square=True)
plt.title(f"Feature Correlation Heatmap (last timestep, {last_timestep.shape[1]} features)")
plt.tight_layout()
plt.savefig("feature_correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()
print("   Saved → feature_correlation_heatmap.png")

print("\nScaler fitting complete.")