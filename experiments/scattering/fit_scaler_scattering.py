# fit_scaler_scattering.py
"""
Fit a StandardScaler on wavelet scattering features extracted from training data.
Analogous to src/data/fit_scaler_V6.py but uses scattering transform instead of
hand-crafted features.
"""

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.scattering.scattering_features import (
    extract_scattering_features, SCATTERING_DIM, WINDOW_SAMPLES, WINDOW_SEC, SAMPLE_RATE
)

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────
STRIDE_SAMPLES = 30
SEQ_LEN = 8  # for filename consistency

OUTPUT_SCALER_PATH = Path(__file__).resolve().parent / f"scattering_scaler_J6_Q8_{SCATTERING_DIM}f.pkl"

TRAINING_DATA_DIR = PROJECT_ROOT / "training_data"

print(f"Scattering Feature Scaler Fitting")
print(f"   Window          = {WINDOW_SEC}s ({WINDOW_SAMPLES} samples)")
print(f"   Stride          = {STRIDE_SAMPLES} samples")
print(f"   Scattering dim  = {SCATTERING_DIM}")
print(f"   Output          = {OUTPUT_SCALER_PATH}")

# ────────────────────────────────────────────────
# Extract features from all training parquets
# ────────────────────────────────────────────────
parquet_files = sorted(TRAINING_DATA_DIR.glob("*_labeled_segment.parquet"))
if not parquet_files:
    print(f"No parquet files found in {TRAINING_DATA_DIR}")
    sys.exit(1)

print(f"\nFound {len(parquet_files)} training files.")

global_features = []

for file_path in tqdm(parquet_files, desc="Processing files"):
    df = pd.read_parquet(file_path, engine='pyarrow')
    
    required_cols = ['timeInMS', 'magRLoadAdjusted', 'label']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"  {file_path.name}: missing {missing} — skipping")
        continue

    df = df[required_cols].copy().dropna().reset_index(drop=True)
    resistance = df['magRLoadAdjusted'].to_numpy(dtype=np.float32)
    labels = df['label'].to_numpy(dtype=np.int64)

    valid_mask = np.isin(labels, [0, 1, 2])

    if len(resistance) < WINDOW_SAMPLES:
        continue

    run_features = []
    skipped = 0

    for start in range(0, len(resistance) - WINDOW_SAMPLES + 1, STRIDE_SAMPLES):
        end = start + WINDOW_SAMPLES
        
        # Skip windows containing unlabeled data
        if not valid_mask[start:end].all():
            skipped += 1
            continue

        window = resistance[start:end]
        feats = extract_scattering_features(window)
        if feats is not None and len(feats) == SCATTERING_DIM:
            run_features.append(feats)

    global_features.extend(run_features)

if not global_features:
    print("No features extracted!")
    sys.exit(1)

X = np.array(global_features, dtype=np.float32)
print(f"\nCollected {X.shape[0]} feature vectors | shape={X.shape}")

X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# ────────────────────────────────────────────────
# Fit scaler
# ────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler

print("Fitting StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nAfter scaling ({X.shape[1]} features):")
print(f"   Mean: {X_scaled.mean():.6f}   (should be ~0)")
print(f"   Std:  {X_scaled.std():.6f}   (should be ~1)")

joblib.dump(scaler, OUTPUT_SCALER_PATH)
print(f"\nScaler saved → {OUTPUT_SCALER_PATH}")
