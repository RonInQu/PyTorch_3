"""
infer_shape.py — Run inference on denoised test data using the
shape_slopes_level model (22 features: 12 original + 10 shape).

Uses the same EMA smoothing, DA override, plotting, and metrics
logic as gru_torch_V6.py process_file, but with a custom feature
extraction pipeline that computes both original and shape features.
"""

import os
import sys
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

import numpy as np
import pandas as pd
import torch
import joblib
from scipy.signal import find_peaks
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.gru_torch_V6 import (
    ClotFeatureExtractor, ClotGRU,
    SEQ_LEN, WINDOW_SEC, REPORT_INTERVAL_MS,
    TEMPERATURE, DEVICE,
    INIT_BLOOD_PROB, INIT_CLOT_PROB, INIT_WALL_PROB,
    DA_LABEL_CONFIDENCE, DA_OTHER_CONFIDENCE,
    GRU_OVERRIDE_THRD_CLOT, GRU_OVERRIDE_THRD_WALL,
    EMA_BLOOD_PRIOR_HISTORY, EMA_BLOOD_PRIOR_NEW,
    EMA_EXIT_TO_BLOOD_HISTORY, EMA_EXIT_TO_BLOOD_NEW,
    EMA_SAME_CLASS_HISTORY, EMA_SAME_CLASS_NEW,
    EMA_CROSS_CLASS_HISTORY, EMA_CROSS_CLASS_NEW,
)

# ── Paths ──
MODEL_PATH  = EXPERIMENT_DIR / "models" / "clot_gru_dn_shape_shape_slopes_level.pt"
SCALER_PATH = EXPERIMENT_DIR / "models" / "scaler_dn_shape_shape_slopes_level.pkl"
TEST_DIR    = EXPERIMENT_DIR / "test_data"
RESULTS_DIR = EXPERIMENT_DIR / "results_shape"
BASELINE_TEST_DIR = PROJECT_ROOT / "test_data"

# ── Feature config for shape_slopes_level (must match training) ──
ORIG_INDICES = [0, 3, 4, 6, 10, 11, 12, 13, 14, 15, 23, 27]  # 12 original features
N_SHAPE = 10
N_FEATURES = len(ORIG_INDICES) + N_SHAPE  # 22

SAMPLE_RATE = 150
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)  # 750


# ════════════════════════════════════════════════════
# SHAPE FEATURE COMPUTATION (from train_shape_features.py)
# ════════════════════════════════════════════════════

def compute_shape_features(window):
    """Compute 10 shape features from a 750-sample window.
    Order: [dir_chg_per_s, peaks_per_s, curvature, longest_mono,
            zero_cross, waveform_len, R2_linear, mean_seg_slope,
            rise_frac, level_change]
    """
    w = window
    n = len(w)
    d = np.diff(w)
    dd = np.diff(d)
    w_std = np.std(w)

    signs = np.sign(d)
    signs[signs == 0] = 1
    dir_changes = np.sum(np.diff(signs) != 0)

    s0 = dir_changes / (n / SAMPLE_RATE)

    if w_std > 0.01:
        peaks, _ = find_peaks(w, prominence=w_std * 0.3)
        s1 = len(peaks) / (n / SAMPLE_RATE)
    else:
        s1 = 0.0

    denom = (1 + d[:-1] ** 2) ** 1.5
    s2 = np.mean(np.abs(dd) / denom)

    changes = np.where(np.diff(signs) != 0)[0]
    runs = np.diff(np.concatenate([[0], changes, [len(signs)]]))
    s3 = np.max(runs) / len(d)

    kernel = 150
    if n > kernel * 2:
        trend = np.convolve(w, np.ones(kernel) / kernel, 'same')
        detr = w[kernel // 2: -(kernel // 2)] - trend[kernel // 2: -(kernel // 2)]
        s4 = np.sum(np.diff(np.sign(detr)) != 0) / max(len(detr), 1)
    else:
        s4 = 0.0

    s5 = np.mean(np.abs(d))

    x = np.arange(n, dtype=np.float64)
    p = np.polyfit(x, w, 1)
    resid = w - np.polyval(p, x)
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((w - w.mean()) ** 2) + 1e-10
    s6 = max(0.0, 1 - ss_res / ss_tot)

    if len(runs) > 1:
        slope_mags = []
        pos = 0
        for rl in runs[:50]:
            seg = w[pos:pos + rl + 1]
            if len(seg) >= 2:
                slope_mags.append(abs(seg[-1] - seg[0]) / len(seg))
            pos += rl
        s7 = np.mean(slope_mags) if slope_mags else 0.0
    else:
        s7 = 0.0

    s8 = np.sum(d > 0) / len(d)

    first_q = np.mean(w[:n // 4])
    last_q = np.mean(w[-n // 4:])
    s9 = (last_q - first_q) / (w_std + 1e-6)

    return np.array([s0, s1, s2, s3, s4, s5, s6, s7, s8, s9], dtype=np.float32)


# ════════════════════════════════════════════════════
# SHAPE-AWARE LIVE DETECTOR
# ════════════════════════════════════════════════════

class ShapeLiveDetector:
    """Like LiveClotDetector but produces 22-feature vectors
    (12 original + 10 shape) for the shape_slopes_level model."""

    def __init__(self):
        self.scaler = joblib.load(SCALER_PATH)
        self.model = ClotGRU(input_size=N_FEATURES).to(DEVICE)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        self.model.eval()

        self.hidden = None
        self.posterior = np.array([INIT_BLOOD_PROB, INIT_CLOT_PROB, INIT_WALL_PROB],
                                  dtype=np.float32)
        self.feat_history = deque(maxlen=SEQ_LEN)
        self.raw_probs = np.array([0.33, 0.33, 0.34], dtype=np.float32)

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
    def predict(self, combined_feats, da_label=None):
        """Run one prediction step with 22-feature vector."""
        scaled = self.scaler.transform(combined_feats.reshape(1, -1))[0]
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
        self.feat_history.append(scaled)

        if len(self.feat_history) < SEQ_LEN:
            pad = list(self.feat_history)[0] if self.feat_history else scaled
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
            alpha_history, alpha_new = EMA_BLOOD_PRIOR_HISTORY, EMA_BLOOD_PRIOR_NEW
        else:
            new_idx = np.argmax(probs)
            if new_idx == 0:
                alpha_history, alpha_new = EMA_EXIT_TO_BLOOD_HISTORY, EMA_EXIT_TO_BLOOD_NEW
            elif new_idx == prior_idx:
                alpha_history, alpha_new = EMA_SAME_CLASS_HISTORY, EMA_SAME_CLASS_NEW
            else:
                alpha_history, alpha_new = EMA_CROSS_CLASS_HISTORY, EMA_CROSS_CLASS_NEW

        self.posterior = alpha_history * self.posterior + alpha_new * probs

        # Post-EMA DA safety net
        if da_label in (1, 2):
            final_idx = np.argmax(self.posterior)
            if final_idx != da_label and self._da_should_override_gru(probs, da_label, strict=True):
                self.posterior = self._make_da_probs(da_label)

        return self.posterior.copy()


# ════════════════════════════════════════════════════
# PROCESS ONE FILE
# ════════════════════════════════════════════════════

def process_file(filepath, all_gt_labels, all_da_labels, all_ml_preds, all_override_times):
    study_name = filepath.stem
    print(f"\nProcessing: {study_name}")

    df = pd.read_parquet(filepath)
    time_ms = df['timeInMS'].values
    resistance = df['magRLoadAdjusted'].values.astype(np.float32)
    # Use baseline GT labels (denoised parquets have modified label column)
    stem_base = study_name.replace('_labeled_segment_denoised', '')
    baseline_path = BASELINE_TEST_DIR / f"{stem_base}_labeled_segment.parquet"
    gt_labels = pd.read_parquet(baseline_path, columns=['label'])['label']
    da_labels = df.get('da_label', None) if 'da_label' in df.columns else None

    # Original feature extractor (computes all 46 features; we pick ORIG_INDICES)
    extractor = ClotFeatureExtractor(active_features=list(range(46)))
    detector = ShapeLiveDetector()

    results = []
    last_report = -REPORT_INTERVAL_MS

    for i, (t, r) in enumerate(zip(time_ms, resistance)):
        extractor.update(float(r))

        if t - last_report >= REPORT_INTERVAL_MS:
            # Get all 46 original features, pick the 12 we need
            all_feats = extractor.compute_features()
            orig_feats = all_feats[ORIG_INDICES]

            # Get shape features from the buffer
            if len(extractor.buffer) >= WINDOW_SAMPLES:
                window_data = np.array(extractor.buffer, dtype=np.float32)
                shape_feats = compute_shape_features(window_data)
            else:
                shape_feats = np.zeros(N_SHAPE, dtype=np.float32)

            # Combine: [12 orig, 10 shape] = 22 features
            combined = np.concatenate([orig_feats, shape_feats])

            da_now = da_labels[i] if da_labels is not None else None
            post = detector.predict(combined, da_now)
            raw = detector.raw_probs
            status = np.argmax(post)
            entropy = -np.sum(post * np.log(post + 1e-12))

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
                'entropy': float(entropy)
            })
            last_report = t

    results_df = pd.DataFrame(results)

    # Save detection_results
    results_df.to_parquet(RESULTS_DIR / f"{study_name}_detection_results.parquet", index=False)
    results_df.to_csv(RESULTS_DIR / f"{study_name}_detection_results.csv", index=False)

    # ── Probability plot ──
    colors = {0: 'black', 1: 'red', 2: 'blue'}
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14), sharex=True)

    for lbl, name in [(0, 'blood'), (1, 'clot'), (2, 'wall')]:
        mask = results_df['prediction'] == lbl
        ax1.scatter(results_df['time'][mask], results_df['resistance'][mask],
                    c=colors[lbl], s=4, label=name, alpha=0.8)
    ax1.set_ylabel('Resistance (Ω)')
    ax1.set_title(f'{study_name} — Detected Labels (shape_slopes_level)')
    ax1.grid(True, alpha=0.3)

    ax2.plot(results_df['time'], results_df['rawC'], color='red', label='raw P(clot)', linewidth=1.2, alpha=0.8)
    ax2.plot(results_df['time'], results_df['rawW'], color='blue', label='raw P(wall)', linewidth=1.2, alpha=0.8)
    ax2.set_ylabel('Probability')
    ax2.set_ylim(0, 1)
    ax2.set_title(f'{study_name} — Raw GRU Probabilities (T={TEMPERATURE})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.plot(results_df['time'], results_df['Cprob'], color='red', label='P(clot)', linewidth=1.8)
    ax3.plot(results_df['time'], results_df['Wprob'], color='blue', label='P(wall)', linewidth=1.8)
    blood_dom = (results_df['Nprob'] > results_df['Cprob']) & (results_df['Nprob'] > results_df['Wprob'])
    ax3.fill_between(results_df['time'], 0, 1, where=blood_dom, color='gray', alpha=0.12, label='Blood dominant')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Probability')
    ax3.set_ylim(0, 1)
    ax3.set_title(f'{study_name} — Smoothed Posterior')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{study_name}_detected_vs_clot_wall_probs.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved probability plot")

    # ── Three-panel plot + metrics ──
    if gt_labels is not None and da_labels is not None:
        gt = gt_labels.values.astype(int)
        da = da_labels.values.astype(int)
        full_times = time_ms / 1000.0

        interp_ml = np.interp(full_times, results_df['time'], results_df['prediction'])
        interp_ml = np.round(interp_ml).astype(int)

        valid = (gt >= 0)
        gt_valid = gt[valid]
        da_valid = da[valid]
        ml_valid = interp_ml[valid]
        n_unlabeled = (~valid).sum()
        if n_unlabeled > 0:
            print(f"  Excluding {n_unlabeled} unlabeled samples (label == -1) from metrics")

        fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True, sharey=True)
        lbl_names = {0: 'blood', 1: 'clot', 2: 'wall'}

        ax = axes[0]
        for lbl in [0, 1, 2]:
            mask = results_df['prediction'] == lbl
            ax.scatter(results_df['time'][mask], results_df['resistance'][mask],
                       c=colors[lbl], s=5, label=lbl_names[lbl], alpha=0.85)

        ml_da = np.interp(results_df['time'], full_times, da)
        ml_da = np.round(ml_da).astype(int)
        diff = (results_df['prediction'].values != ml_da)
        diff_diff = np.diff(diff.astype(int))
        starts = np.where(diff_diff == 1)[0] + 1
        ends = np.where(diff_diff == -1)[0] + 1
        if diff[0]:
            starts = np.insert(starts, 0, 0)
        if diff[-1]:
            ends = np.append(ends, len(diff))
        for s, e in zip(starts, ends):
            ax.axvspan(results_df['time'].iloc[s], results_df['time'].iloc[e - 1],
                       facecolor='#e8e8e8', alpha=0.55, label='ML ≠ DA' if s == starts[0] else None)

        ax.set_title(f'{study_name} — ML Predictions (shape_slopes_level)')
        ax.set_ylabel('Resistance (Ω)')
        ax.grid(True, alpha=0.3)

        for ax_idx, (title, data) in enumerate([
            ("DA Labels (full 150 Hz)", da),
            ("Ground Truth Labels (full 150 Hz)", gt)
        ]):
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
            ax.grid(True, alpha=0.3)
            if ax_idx == 1:
                ax.set_xlabel('Time (seconds)')

        plt.tight_layout(h_pad=0.8)
        plt.savefig(RESULTS_DIR / f"{study_name}_ml_da_gt_three_panel.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved three-panel plot")

        # Metrics
        print(f"\n{study_name} metrics:")
        print(f"DA  Acc: {accuracy_score(gt_valid, da_valid):.4f}  "
              f"F1: {f1_score(gt_valid, da_valid, average='macro'):.4f}  "
              f"Prec: {precision_score(gt_valid, da_valid, average='macro'):.4f}  "
              f"Rec: {recall_score(gt_valid, da_valid, average='macro'):.4f}")
        print(f"ML  Acc: {accuracy_score(gt_valid, ml_valid):.4f}  "
              f"F1: {f1_score(gt_valid, ml_valid, average='macro'):.4f}  "
              f"Prec: {precision_score(gt_valid, ml_valid, average='macro'):.4f}  "
              f"Rec: {recall_score(gt_valid, ml_valid, average='macro'):.4f}")
        print(f"Improvement: Acc {accuracy_score(gt_valid, ml_valid) - accuracy_score(gt_valid, da_valid):+.4f}   "
              f"F1 {f1_score(gt_valid, ml_valid, average='macro') - f1_score(gt_valid, da_valid, average='macro'):+.4f}")

        # Override analysis
        override_mask = (ml_valid != da_valid)
        n_overrides = override_mask.sum()
        if n_overrides > 0:
            correct = (ml_valid[override_mask] == gt_valid[override_mask]).sum()
            harmful = (da_valid[override_mask] == gt_valid[override_mask]).sum()
            override_prec = correct / n_overrides
            da_cw_errors = ((da_valid != gt_valid) & ((gt_valid == 1) | (gt_valid == 2))).sum()
            override_rec = correct / da_cw_errors if da_cw_errors > 0 else 0.0

            print(f"\n  Override analysis ({study_name}):")
            print(f"    Total overrides:   {n_overrides}")
            print(f"    Correct (ML right, DA wrong): {correct}")
            print(f"    Harmful (DA right, ML wrong): {harmful}")
            print(f"    Override Precision: {override_prec:.4f}")
            print(f"    Override Recall:    {override_rec:.4f}  (of {da_cw_errors} DA clot/wall errors)")
            print(f"    Net benefit:        {correct - harmful:+d}")
        else:
            print(f"\n  No overrides in {study_name}")

        all_gt_labels.extend(gt_valid)
        all_da_labels.extend(da_valid)
        all_ml_preds.extend(ml_valid)

        overrides = np.where(ml_valid != da_valid)[0]
        all_override_times.extend(full_times[valid][overrides])

    print(f"Finished {study_name}\n")


# ════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(TEST_DIR.glob("*_labeled_segment_denoised.parquet"))

    print("=" * 70)
    print("INFERENCE — SHAPE_SLOPES_LEVEL (22 features: 12 orig + 10 shape)")
    print("=" * 70)
    print(f"  Model:    {MODEL_PATH}")
    print(f"  Scaler:   {SCALER_PATH}")
    print(f"  Test dir: {TEST_DIR}")
    print(f"  Output:   {RESULTS_DIR}")
    print(f"  Files:    {len(files)}")
    print("-" * 70)

    if not files:
        print(f"No denoised parquet files in {TEST_DIR}")
        sys.exit(1)

    all_gt_labels = []
    all_da_labels = []
    all_ml_preds = []
    all_override_times = []

    for f in files:
        process_file(f, all_gt_labels, all_da_labels, all_ml_preds, all_override_times)

    # ── Global summary ──
    if all_gt_labels:
        print("\n" + "=" * 70)
        print("GLOBAL SUMMARY — SHAPE_SLOPES_LEVEL ON DENOISED TEST DATA")
        print("=" * 70)

        acc_da = accuracy_score(all_gt_labels, all_da_labels)
        f1_da = f1_score(all_gt_labels, all_da_labels, average='macro', zero_division=0)
        prec_da = precision_score(all_gt_labels, all_da_labels, average='macro', zero_division=0)
        rec_da = recall_score(all_gt_labels, all_da_labels, average='macro', zero_division=0)

        acc_ml = accuracy_score(all_gt_labels, all_ml_preds)
        f1_ml = f1_score(all_gt_labels, all_ml_preds, average='macro', zero_division=0)
        prec_ml = precision_score(all_gt_labels, all_ml_preds, average='macro', zero_division=0)
        rec_ml = recall_score(all_gt_labels, all_ml_preds, average='macro', zero_division=0)

        print(f"DA  Accuracy: {acc_da:.4f}    F1-macro: {f1_da:.4f}")
        print(f"ML  Accuracy: {acc_ml:.4f}    F1-macro: {f1_ml:.4f}")
        print(f"Improvement: Acc {acc_ml - acc_da:+.4f}   F1 {f1_ml - f1_da:+.4f}")
        print(f"DA  Precision: {prec_da:.4f}    Recall: {rec_da:.4f}")
        print(f"ML  Precision: {prec_ml:.4f}    Recall: {rec_ml:.4f}")

        gt_arr = np.array(all_gt_labels)
        da_arr = np.array(all_da_labels)
        ml_arr = np.array(all_ml_preds)

        g_override_mask = (ml_arr != da_arr)
        g_n_overrides = g_override_mask.sum()

        print(f"\n{'─' * 70}")
        print(f"GLOBAL OVERRIDE ANALYSIS")
        print(f"{'─' * 70}")
        print(f"Total overrides across all studies: {g_n_overrides}")

        if g_n_overrides > 0:
            g_correct = (ml_arr[g_override_mask] == gt_arr[g_override_mask]).sum()
            g_harmful = (da_arr[g_override_mask] == gt_arr[g_override_mask]).sum()
            g_neither = g_n_overrides - g_correct - g_harmful
            g_override_prec = g_correct / g_n_overrides
            g_da_cw_errors = ((da_arr != gt_arr) & ((gt_arr == 1) | (gt_arr == 2))).sum()
            g_override_rec = g_correct / g_da_cw_errors if g_da_cw_errors > 0 else 0.0

            print(f"  Correct overrides (ML right, DA wrong): {g_correct}")
            print(f"  Harmful overrides (DA right, ML wrong): {g_harmful}")
            print(f"  Neither correct (both wrong differently): {g_neither}")
            print(f"  Override Precision: {g_override_prec:.4f}")
            print(f"  Override Recall:    {g_override_rec:.4f}  (of {g_da_cw_errors} DA clot/wall errors)")
            print(f"  Net benefit:        {g_correct - g_harmful:+d} samples")

        # Reference baselines for comparison
        print(f"\n{'─' * 70}")
        print(f"REFERENCE BASELINES")
        print(f"{'─' * 70}")
        print(f"  Original baseline (non-denoised, CWF 21 features):")
        print(f"    F1-macro: 0.6665, Net benefit: +24,638")
        print(f"  Denoised CWF baseline (21 features):")
        print(f"    F1-macro: 0.6099, Net benefit: -107,153")

    print("\nDone.")


if __name__ == "__main__":
    main()
