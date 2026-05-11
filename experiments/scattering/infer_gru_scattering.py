# infer_gru_scattering.py
"""
Inference script for Scattering GRU model.
Same EMA posterior + DA override logic as gru_torch_V6.py, but uses
wavelet scattering features instead of hand-crafted features.

Processes test_data/ parquets, produces probability plots and metrics.
"""

import os
import sys
import warnings
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.scattering.scattering_features import (
    extract_scattering_features, SCATTERING_DIM, WINDOW_SAMPLES, WINDOW_SEC, SAMPLE_RATE
)
from experiments.scattering.train_gru_scattering import ScatteringGRU

# ────────────────────────────────────────────────
# CONFIG (same as gru_torch_V6)
# ────────────────────────────────────────────────
SEQ_LEN = 8
REPORT_INTERVAL_MS = 200
TEMPERATURE = 1.5

GRU_OVERRIDE_THRD_CLOT = 0.80
GRU_OVERRIDE_THRD_WALL = 0.92

EMA_BLOOD_PRIOR_HISTORY = 0.78
EMA_EXIT_TO_BLOOD_HISTORY = 0.35
EMA_SAME_CLASS_HISTORY = 0.97
EMA_CROSS_CLASS_HISTORY = 0.99

DA_LABEL_CONFIDENCE = 0.92
DA_OTHER_CONFIDENCE = (1.0 - DA_LABEL_CONFIDENCE) / 2

INIT_BLOOD_PROB = 0.95

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "models" / "scattering_gru_trained.pt"
SCALER_PATH = SCRIPT_DIR / f"scattering_scaler_J6_Q8_{SCATTERING_DIM}f.pkl"
TEST_DIR = PROJECT_ROOT / "test_data"
OUTPUT_FOLDER = SCRIPT_DIR / "results"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# ────────────────────────────────────────────────
# Live Detector (same posterior logic as gru_torch_V6)
# ────────────────────────────────────────────────
class ScatteringLiveDetector:
    def __init__(self):
        self.model = ScatteringGRU(input_dim=SCATTERING_DIM).to(DEVICE)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        self.model.eval()

        self.scaler = joblib.load(SCALER_PATH)
        self.reset()

    def reset(self):
        self.hidden = None
        self.posterior = np.array([INIT_BLOOD_PROB,
                                   (1-INIT_BLOOD_PROB)/2,
                                   (1-INIT_BLOOD_PROB)/2], dtype=np.float32)
        self.feat_history = deque(maxlen=SEQ_LEN)
        self.raw_probs = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    def _make_da_probs(self, da_label):
        da_probs = np.array([DA_OTHER_CONFIDENCE] * 3, dtype=np.float32)
        da_probs[da_label] = DA_LABEL_CONFIDENCE
        return da_probs

    def _da_should_override_gru(self, probs, da_label, strict=False):
        gru_top_idx = np.argmax(probs)
        if gru_top_idx == da_label:
            return False
        threshold = GRU_OVERRIDE_THRD_CLOT if da_label == 1 else GRU_OVERRIDE_THRD_WALL
        if strict:
            return probs[gru_top_idx] < threshold
        return probs[gru_top_idx] <= threshold

    @torch.no_grad()
    def predict(self, scattering_feats, da_label=None):
        """Run one prediction step with scattering features."""
        scaled = self.scaler.transform(scattering_feats.reshape(1, -1))[0]
        self.feat_history.append(scaled)

        if len(self.feat_history) < SEQ_LEN:
            pad = list(self.feat_history)[0]
            seq_list = [pad] * (SEQ_LEN - len(self.feat_history)) + list(self.feat_history)
        else:
            seq_list = list(self.feat_history)

        seq = np.array(seq_list, dtype=np.float32)
        x = torch.from_numpy(seq).float().unsqueeze(0).to(DEVICE)

        logits, self.hidden = self.model(x, self.hidden)
        if self.hidden is not None:
            self.hidden = self.hidden.detach()

        probs = torch.softmax(logits / TEMPERATURE, 1).squeeze(0).cpu().numpy()
        self.raw_probs = probs.copy()

        prior_idx = np.argmax(self.posterior)

        # DA override
        if da_label is not None:
            if da_label == 0:
                self.posterior = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                self.hidden = None
                self.feat_history.clear()
                return self.posterior.copy()
            elif da_label in (1, 2):
                if self._da_should_override_gru(probs, da_label):
                    probs = self._make_da_probs(da_label)

        # EMA blending
        if prior_idx == 0:
            alpha_h = EMA_BLOOD_PRIOR_HISTORY
        else:
            new_idx = np.argmax(probs)
            if new_idx == 0:
                alpha_h = EMA_EXIT_TO_BLOOD_HISTORY
            elif new_idx == prior_idx:
                alpha_h = EMA_SAME_CLASS_HISTORY
            else:
                alpha_h = EMA_CROSS_CLASS_HISTORY

        self.posterior = alpha_h * self.posterior + (1 - alpha_h) * probs

        # Post-EMA safety
        if da_label in (1, 2):
            final_idx = np.argmax(self.posterior)
            if final_idx != da_label and self._da_should_override_gru(probs, da_label, strict=True):
                self.posterior = self._make_da_probs(da_label)

        return self.posterior.copy()


# ────────────────────────────────────────────────
# Streaming feature extraction from raw resistance
# ────────────────────────────────────────────────
class StreamingScatteringExtractor:
    """Buffers resistance samples and extracts scattering features per window."""
    def __init__(self):
        self.buffer = deque(maxlen=WINDOW_SAMPLES)
        self.ready = False

    def update(self, r_value):
        self.buffer.append(r_value)
        if len(self.buffer) == WINDOW_SAMPLES:
            self.ready = True

    def compute_features(self):
        if not self.ready:
            return None
        window = np.array(self.buffer, dtype=np.float32)
        return extract_scattering_features(window)


# ────────────────────────────────────────────────
# Process a single test file
# ────────────────────────────────────────────────
def process_file(filepath, all_gt_labels, all_da_labels, all_ml_preds):
    study_name = filepath.stem
    print(f"\nProcessing: {study_name}")

    df = pd.read_parquet(filepath)
    time_ms = df['timeInMS'].values
    resistance = df['magRLoadAdjusted'].values.astype(np.float32)
    gt_labels = df['label'].values.astype(int) if 'label' in df.columns else None
    da_labels = df['da_label'].values.astype(int) if 'da_label' in df.columns else None

    extractor = StreamingScatteringExtractor()
    detector = ScatteringLiveDetector()

    results = []
    last_report = -REPORT_INTERVAL_MS

    for i, (t, r) in enumerate(zip(time_ms, resistance)):
        extractor.update(float(r))

        if t - last_report >= REPORT_INTERVAL_MS and extractor.ready:
            feats = extractor.compute_features()
            da_now = int(da_labels[i]) if da_labels is not None else None
            post = detector.predict(feats, da_now)
            raw = detector.raw_probs
            status = np.argmax(post)

            results.append({
                'time': t / 1000.0,
                'prediction': status,
                'resistance': float(r),
                'Nprob': float(post[0]),
                'Cprob': float(post[1]),
                'Wprob': float(post[2]),
                'rawN': float(raw[0]),
                'rawC': float(raw[1]),
                'rawW': float(raw[2]),
            })
            last_report = t

    results_df = pd.DataFrame(results)

    # Plots
    colors = {0: 'black', 1: 'red', 2: 'blue'}
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14), sharex=True)

    for lbl, name in [(0, 'blood'), (1, 'clot'), (2, 'wall')]:
        mask = results_df['prediction'] == lbl
        ax1.scatter(results_df['time'][mask], results_df['resistance'][mask],
                    c=colors[lbl], s=4, label=name, alpha=0.8)
    ax1.set_ylabel('Resistance (Ω)')
    ax1.set_title(f'{study_name} — Scattering GRU Predictions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(results_df['time'], results_df['rawC'], 'r-', label='raw P(clot)', lw=1.2, alpha=0.8)
    ax2.plot(results_df['time'], results_df['rawW'], 'b-', label='raw P(wall)', lw=1.2, alpha=0.8)
    ax2.set_ylabel('Probability')
    ax2.set_ylim(0, 1)
    ax2.set_title(f'{study_name} — Raw GRU Probs (T={TEMPERATURE})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.plot(results_df['time'], results_df['Cprob'], 'r-', label='P(clot)', lw=1.8)
    ax3.plot(results_df['time'], results_df['Wprob'], 'b-', label='P(wall)', lw=1.8)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Probability')
    ax3.set_ylim(0, 1)
    ax3.set_title(f'{study_name} — Smoothed Posterior')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / f"{study_name}_scattering_gru_probs.png", dpi=250, bbox_inches='tight')
    plt.close()

    # Metrics
    if gt_labels is not None and da_labels is not None:
        full_times = time_ms / 1000.0
        interp_ml = np.interp(full_times, results_df['time'], results_df['prediction'])
        interp_ml = np.round(interp_ml).astype(int)

        valid = gt_labels >= 0
        gt_v = gt_labels[valid]
        da_v = da_labels[valid]
        ml_v = interp_ml[valid]

        all_gt_labels.extend(gt_v.tolist())
        all_da_labels.extend(da_v.tolist())
        all_ml_preds.extend(ml_v.tolist())

        da_f1 = f1_score(gt_v, da_v, average='macro')
        ml_f1 = f1_score(gt_v, ml_v, average='macro')
        print(f"  DA F1={da_f1:.4f}  ML F1={ml_f1:.4f}  Δ={ml_f1-da_f1:+.4f}")

        # Override analysis
        override_mask = ml_v != da_v
        n_ov = override_mask.sum()
        if n_ov > 0:
            correct = (ml_v[override_mask] == gt_v[override_mask]).sum()
            harmful = (da_v[override_mask] == gt_v[override_mask]).sum()
            print(f"  Overrides: {n_ov} total, {correct} correct, {harmful} harmful")


# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  SCATTERING GRU INFERENCE")
    print("=" * 60)
    print(f"  Model:   {MODEL_PATH}")
    print(f"  Scaler:  {SCALER_PATH}")
    print(f"  Test:    {TEST_DIR}")
    print(f"  Output:  {OUTPUT_FOLDER}")
    print()

    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)
    if not SCALER_PATH.exists():
        print(f"ERROR: Scaler not found at {SCALER_PATH}")
        sys.exit(1)

    test_files = sorted(TEST_DIR.glob("*_labeled_segment.parquet"))
    if not test_files:
        print(f"No test files in {TEST_DIR}")
        sys.exit(1)

    print(f"Found {len(test_files)} test files.\n")

    all_gt, all_da, all_ml = [], [], []

    for f in test_files:
        process_file(f, all_gt, all_da, all_ml)

    # Global summary
    if all_gt:
        gt = np.array(all_gt)
        da = np.array(all_da)
        ml = np.array(all_ml)

        print("\n" + "=" * 60)
        print("GLOBAL SUMMARY")
        print("=" * 60)
        da_f1 = f1_score(gt, da, average='macro')
        ml_f1 = f1_score(gt, ml, average='macro')
        da_acc = accuracy_score(gt, da)
        ml_acc = accuracy_score(gt, ml)
        print(f"DA  Acc={da_acc:.4f}  F1={da_f1:.4f}")
        print(f"ML  Acc={ml_acc:.4f}  F1={ml_f1:.4f}")
        print(f"Improvement: Acc {ml_acc-da_acc:+.4f}  F1 {ml_f1-da_f1:+.4f}")

        override_mask = ml != da
        n_ov = override_mask.sum()
        correct = (ml[override_mask] == gt[override_mask]).sum()
        harmful = (da[override_mask] == gt[override_mask]).sum()
        net = correct - harmful
        print(f"\nOverrides: {n_ov} total")
        print(f"  Correct: {correct}  Harmful: {harmful}  Net: {net:+d}")
        if n_ov > 0:
            print(f"  Override precision: {correct/n_ov:.4f}")

        # Save summary
        with open(OUTPUT_FOLDER / "global_summary.txt", 'w') as f:
            f.write(f"Scattering GRU Global Summary\n")
            f.write(f"DA  Acc={da_acc:.4f}  F1={da_f1:.4f}\n")
            f.write(f"ML  Acc={ml_acc:.4f}  F1={ml_f1:.4f}\n")
            f.write(f"Improvement: Acc {ml_acc-da_acc:+.4f}  F1 {ml_f1-da_f1:+.4f}\n")
            f.write(f"Overrides: {n_ov}, Correct: {correct}, Harmful: {harmful}, Net: {net:+d}\n")


if __name__ == '__main__':
    main()
