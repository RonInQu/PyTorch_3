"""
fit_scaler_dn.py — Fit StandardScaler on denoised training features.

Identical logic to fit_scaler_V6.py but reads from experiments/denoised/train_data/
and saves scaler to experiments/denoised/models/.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from scipy.signal import lfilter

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Import config from existing codebase (single source of truth) ──
from src.models.gru_torch_V6 import (
    FEATURE_SET, TOTAL_FEATURES, ClotFeatureExtractor,
    SEQ_LEN, WINDOW_SEC, active_idx, active_dim, dim_str,
)
from src.training.train_gru_V6 import STRIDE_SAMPLES

# ── Paths (denoised experiment) ──
SAMPLE_RATE = 150
TRAINING_DATA_DIR = EXPERIMENT_DIR / "train_data"
MODELS_DIR = EXPERIMENT_DIR / "models"
OUTPUT_SCALER_PATH = MODELS_DIR / "scaler_denoised.pkl"

# ── Extractor + lfilter setup ──
extractor = ClotFeatureExtractor(sample_rate=SAMPLE_RATE, window_sec=WINDOW_SEC,
                                 active_features=active_idx)
WINDOW_SAMPLES = extractor.window_size
ALPHA_FAST = extractor.alpha_fast
ALPHA_SLOW = extractor.alpha_slow

_B_FAST = np.array([ALPHA_FAST])
_A_FAST = np.array([1.0, -(1.0 - ALPHA_FAST)])
_B_SLOW = np.array([ALPHA_SLOW])
_A_SLOW = np.array([1.0, -(1.0 - ALPHA_SLOW)])


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FIT SCALER — DENOISED EXPERIMENT")
    print("=" * 60)
    print(f"  Feature set:     {FEATURE_SET} ({active_dim} features)")
    print(f"  SEQ_LEN:         {SEQ_LEN}")
    print(f"  Window:          {WINDOW_SEC}s ({WINDOW_SAMPLES} samples)")
    print(f"  Stride:          {STRIDE_SAMPLES}")
    print(f"  Training data:   {TRAINING_DATA_DIR}")
    print(f"  Output scaler:   {OUTPUT_SCALER_PATH}")
    print("-" * 60)

    parquet_files = sorted(TRAINING_DATA_DIR.glob("*_labeled_segment_denoised.parquet"))
    if not parquet_files:
        print(f"No denoised parquet files found in {TRAINING_DATA_DIR}")
        sys.exit(1)

    print(f"Found {len(parquet_files)} denoised training files.\n")

    global_features = []

    for file_path in tqdm(parquet_files, desc="Extracting features"):
        df = pd.read_parquet(file_path, engine='pyarrow')

        required_cols = ['timeInMS', 'magRLoadAdjusted', 'label']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"  {file_path.name}: missing {missing} — skipping")
            continue

        df = df[required_cols].copy().dropna().reset_index(drop=True)
        resistance = df['magRLoadAdjusted'].to_numpy(dtype=np.float32)
        labels = df['label'].to_numpy(dtype=np.int64)

        valid_mask = np.isin(labels, [0, 1, 2])

        if len(resistance) < SAMPLE_RATE * WINDOW_SEC:
            continue

        run_features = []
        for start in range(0, len(resistance) - WINDOW_SAMPLES + 1, STRIDE_SAMPLES):
            if not valid_mask[start : start + WINDOW_SAMPLES].all():
                continue

            window_res = resistance[start : start + WINDOW_SAMPLES]

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

    if not global_features:
        print("No features extracted!")
        sys.exit(1)

    X = np.array(global_features, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"\nCollected {X.shape[0]} feature vectors | shape={X.shape}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\nAfter scaling ({X.shape[1]} features):")
    print(f"   Mean: {X_scaled.mean():.6f}  (target ~0)")
    print(f"   Std:  {X_scaled.std():.6f}  (target ~1)")

    joblib.dump(scaler, OUTPUT_SCALER_PATH)
    print(f"\n✅ Scaler saved → {OUTPUT_SCALER_PATH}")


if __name__ == "__main__":
    main()
