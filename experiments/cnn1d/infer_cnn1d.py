# infer_cnn1d.py
"""
Inference script for 1D CNN (ResNet) on raw waveforms.
Processes test_data/ parquets with EMA posterior smoothing + DA override.

Operates in streaming mode: buffers 750 samples, classifies each window,
smooths via EMA posterior — same logic as gru_torch_V6.
"""

import os
import sys
import warnings
from collections import deque
from pathlib import Path

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.cnn1d.train_cnn1d import ClotCNN1D, WINDOW_SAMPLES, WINDOW_SEC, SAMPLE_RATE

# ────────────────────────────────────────────────
# CONFIG (matching gru_torch_V6 inference settings)
# ────────────────────────────────────────────────
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

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "models" / "cnn1d_trained.pt"
TEST_DIR = PROJECT_ROOT / "test_data"
OUTPUT_FOLDER = SCRIPT_DIR / "results"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# ────────────────────────────────────────────────
# Live Detector with CNN
# ────────────────────────────────────────────────
class CNN1DLiveDetector:
    def __init__(self):
        self.model = ClotCNN1D().to(DEVICE)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        self.model.eval()
        self.reset()

    def reset(self):
        self.posterior = np.array([INIT_BLOOD_PROB,
                                   (1 - INIT_BLOOD_PROB) / 2,
                                   (1 - INIT_BLOOD_PROB) / 2], dtype=np.float32)
        self.raw_probs = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    def _make_da_probs(self, da_label):
        da_probs = np.array([DA_OTHER_CONFIDENCE] * 3, dtype=np.float32)
        da_probs[da_label] = DA_LABEL_CONFIDENCE
        return da_probs

    def _da_should_override(self, probs, da_label, strict=False):
        top_idx = np.argmax(probs)
        if top_idx == da_label:
            return False
        threshold = GRU_OVERRIDE_THRD_CLOT if da_label == 1 else GRU_OVERRIDE_THRD_WALL
        if strict:
            return probs[top_idx] < threshold
        return probs[top_idx] <= threshold

    @torch.no_grad()
    def predict(self, window, da_label=None):
        """
        Classify a raw 750-sample window.
        Returns 3-element posterior [P(blood), P(clot), P(wall)].
        """
        x = torch.from_numpy(window).float().unsqueeze(0).to(DEVICE)
        logits = self.model(x)
        probs = torch.softmax(logits / TEMPERATURE, 1).squeeze(0).cpu().numpy()
        self.raw_probs = probs.copy()

        prior_idx = np.argmax(self.posterior)

        # DA override
        if da_label is not None:
            if da_label == 0:
                self.posterior = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                return self.posterior.copy()
            elif da_label in (1, 2):
                if self._da_should_override(probs, da_label):
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
            if final_idx != da_label and self._da_should_override(probs, da_label, strict=True):
                self.posterior = self._make_da_probs(da_label)

        return self.posterior.copy()


# ────────────────────────────────────────────────
# Process a single test file
# ────────────────────────────────────────────────
def process_file(filepath, all_gt, all_da, all_ml):
    study_name = filepath.stem
    print(f"\nProcessing: {study_name}")

    df = pd.read_parquet(filepath)
    time_ms = df['timeInMS'].values
    resistance = df['magRLoadAdjusted'].values.astype(np.float32)
    gt_labels = df['label'].values.astype(int) if 'label' in df.columns else None
    da_labels = df['da_label'].values.astype(int) if 'da_label' in df.columns else None

    detector = CNN1DLiveDetector()
    results = []
    last_report = -REPORT_INTERVAL_MS

    for i in range(len(time_ms)):
        t = time_ms[i]
        if t - last_report < REPORT_INTERVAL_MS:
            continue
        if i < WINDOW_SAMPLES - 1:
            continue

        last_report = t

        # Extract raw window
        window = resistance[i - WINDOW_SAMPLES + 1: i + 1]
        da_now = int(da_labels[i]) if da_labels is not None else None
        post = detector.predict(window, da_now)
        raw = detector.raw_probs
        status = np.argmax(post)

        results.append({
            'time': t / 1000.0,
            'prediction': status,
            'resistance': float(resistance[i]),
            'Nprob': float(post[0]),
            'Cprob': float(post[1]),
            'Wprob': float(post[2]),
            'rawN': float(raw[0]),
            'rawC': float(raw[1]),
            'rawW': float(raw[2]),
        })

    results_df = pd.DataFrame(results)

    # ── Probability plot ──
    colors = {0: 'black', 1: 'red', 2: 'blue'}
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14), sharex=True)

    for lbl, name in [(0, 'blood'), (1, 'clot'), (2, 'wall')]:
        mask = results_df['prediction'] == lbl
        ax1.scatter(results_df['time'][mask], results_df['resistance'][mask],
                    c=colors[lbl], s=4, label=name, alpha=0.8)
    ax1.set_ylabel('Resistance (Ω)')
    ax1.set_title(f'{study_name} — 1D CNN Predictions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(results_df['time'], results_df['rawC'], 'r-', label='raw P(clot)', lw=1.2, alpha=0.8)
    ax2.plot(results_df['time'], results_df['rawW'], 'b-', label='raw P(wall)', lw=1.2, alpha=0.8)
    ax2.set_ylabel('Probability')
    ax2.set_ylim(0, 1)
    ax2.set_title(f'{study_name} — Raw CNN Probs (T={TEMPERATURE})')
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
    plt.savefig(OUTPUT_FOLDER / f"{study_name}_cnn1d_probs.png", dpi=250, bbox_inches='tight')
    plt.close()

    # ── Metrics + three-panel plot ──
    if gt_labels is not None and da_labels is not None:
        full_times = time_ms / 1000.0
        interp_ml = np.interp(full_times, results_df['time'], results_df['prediction'])
        interp_ml = np.round(interp_ml).astype(int)

        valid = gt_labels >= 0
        gt_v = gt_labels[valid]
        da_v = da_labels[valid]
        ml_v = interp_ml[valid]

        all_gt.extend(gt_v.tolist())
        all_da.extend(da_v.tolist())
        all_ml.extend(ml_v.tolist())

        da_f1 = f1_score(gt_v, da_v, average='macro')
        ml_f1 = f1_score(gt_v, ml_v, average='macro')
        print(f"  DA F1={da_f1:.4f}  ML F1={ml_f1:.4f}  Δ={ml_f1-da_f1:+.4f}")

        override_mask = ml_v != da_v
        n_ov = override_mask.sum()
        if n_ov > 0:
            correct = (ml_v[override_mask] == gt_v[override_mask]).sum()
            harmful = (da_v[override_mask] == gt_v[override_mask]).sum()
            print(f"  Overrides: {n_ov}, correct={correct}, harmful={harmful}, net={correct-harmful:+d}")

        # ── Three-panel plot: ML (top), DA (middle), GT (bottom) ──
        lbl_names = {0: 'blood', 1: 'clot', 2: 'wall'}
        fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True, sharey=True,
                                 gridspec_kw={'height_ratios': [1, 1, 1]})

        # Top: ML predictions
        ax = axes[0]
        for lbl in [0, 1, 2]:
            mask = results_df['prediction'] == lbl
            ax.scatter(results_df['time'][mask], results_df['resistance'][mask],
                       c=colors[lbl], s=5, label=lbl_names[lbl], alpha=0.85)

        # Highlight override regions (ML ≠ DA)
        ml_da_interp = np.interp(results_df['time'], full_times, da_labels)
        ml_da_interp = np.round(ml_da_interp).astype(int)
        diff = (results_df['prediction'].values != ml_da_interp)
        diff_diff = np.diff(diff.astype(int))
        starts = np.where(diff_diff == 1)[0] + 1
        ends = np.where(diff_diff == -1)[0] + 1
        if diff.size > 0 and diff[0]:
            starts = np.insert(starts, 0, 0)
        if diff.size > 0 and diff[-1]:
            ends = np.append(ends, len(diff))
        for s, e in zip(starts, ends):
            ax.axvspan(results_df['time'].iloc[s], results_df['time'].iloc[min(e-1, len(results_df)-1)],
                       facecolor='#e8e8e8', alpha=0.55,
                       label='ML ≠ DA' if s == starts[0] else None)

        ax.set_title(f'{study_name} — ML Predictions (1D CNN, 200 ms reporting)')
        ax.set_ylabel('Resistance (Ω)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Middle: DA labels, Bottom: GT labels
        panel_data = [("DA Labels (full 150 Hz)", da_labels),
                      ("Ground Truth Labels (full 150 Hz)", gt_labels)]
        for ax_idx, (title, data) in enumerate(panel_data):
            ax = axes[ax_idx + 1]
            unlabeled_mask = data == -1
            if unlabeled_mask.any():
                ax.scatter(full_times[unlabeled_mask], resistance[unlabeled_mask],
                           c='black', s=2, label='unlabeled', alpha=0.4, zorder=1)
            for lbl in [0, 1, 2]:
                mask = data == lbl
                ax.scatter(full_times[mask], resistance[mask], c=colors[lbl], s=2,
                           label=lbl_names[lbl], alpha=0.7, zorder=2)
            ax.set_title(f'{study_name} — {title}')
            ax.set_ylabel('Resistance (Ω)')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            if ax_idx == 1:
                ax.set_xlabel('Time (seconds)')

        plt.tight_layout(h_pad=0.8)
        plt.savefig(OUTPUT_FOLDER / f"{study_name}_ml_da_gt_three_panel.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved three-panel plot")


# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  1D CNN (ResNet) INFERENCE")
    print("=" * 60)
    print(f"  Model:  {MODEL_PATH}")
    print(f"  Test:   {TEST_DIR}")
    print(f"  Output: {OUTPUT_FOLDER}")
    print()

    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)

    test_files = sorted(TEST_DIR.glob("*_labeled_segment.parquet"))
    if not test_files:
        print(f"No test files in {TEST_DIR}")
        sys.exit(1)

    print(f"Found {len(test_files)} test files.\n")

    all_gt, all_da, all_ml = [], [], []

    for f in test_files:
        process_file(f, all_gt, all_da, all_ml)

    # ── Global summary ──
    if all_gt:
        gt = np.array(all_gt)
        da = np.array(all_da)
        ml = np.array(all_ml)

        print("\n" + "=" * 60)
        print("GLOBAL SUMMARY — 1D CNN")
        print("=" * 60)
        da_f1 = f1_score(gt, da, average='macro')
        ml_f1 = f1_score(gt, ml, average='macro')
        da_acc = accuracy_score(gt, da)
        ml_acc = accuracy_score(gt, ml)
        da_prec = precision_score(gt, da, average='macro')
        ml_prec = precision_score(gt, ml, average='macro')
        da_rec = recall_score(gt, da, average='macro')
        ml_rec = recall_score(gt, ml, average='macro')

        print(f"DA  Accuracy: {da_acc:.4f}    F1-macro: {da_f1:.4f}")
        print(f"ML  Accuracy: {ml_acc:.4f}    F1-macro: {ml_f1:.4f}")
        print(f"Improvement: Acc {ml_acc-da_acc:+.4f}   F1 {ml_f1-da_f1:+.4f}")
        print()
        print(f"DA  Precision: {da_prec:.4f}    Recall: {da_rec:.4f}")
        print(f"ML  Precision: {ml_prec:.4f}    Recall: {ml_rec:.4f}")

        override_mask = ml != da
        n_ov = override_mask.sum()
        correct = (ml[override_mask] == gt[override_mask]).sum()
        harmful = (da[override_mask] == gt[override_mask]).sum()
        net = correct - harmful

        print(f"\nOverride Analysis:")
        print(f"  Total overrides: {n_ov}")
        print(f"  Correct: {correct}  Harmful: {harmful}")
        print(f"  Net benefit: {net:+d} samples")
        if n_ov > 0:
            print(f"  Override precision: {correct/n_ov:.4f}")

        summary_path = OUTPUT_FOLDER / "global_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("GLOBAL SUMMARY — 1D CNN (ResNet) on Raw Waveforms\n")
            f.write(f"DA  Accuracy: {da_acc:.4f}    F1-macro: {da_f1:.4f}\n")
            f.write(f"ML  Accuracy: {ml_acc:.4f}    F1-macro: {ml_f1:.4f}\n")
            f.write(f"Improvement: Acc {ml_acc-da_acc:+.4f}   F1 {ml_f1-da_f1:+.4f}\n\n")
            f.write(f"Overrides: {n_ov}, Correct: {correct}, Harmful: {harmful}, Net: {net:+d}\n")
            if n_ov > 0:
                f.write(f"Override Precision: {correct/n_ov:.4f}\n")
        print(f"\nSaved → {summary_path}")


if __name__ == '__main__':
    main()
