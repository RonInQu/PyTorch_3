# gru_torch_V5.py
"""
Real-time clot detection — V5 (cleaned, REDUCE_DIM compatible version of working V2)
"""

import os
import warnings
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ────────────────────────────────────────────────
# CONFIG — Single source of truth
# ────────────────────────────────────────────────
SEQ_LEN = 8

REDUCE_DIM = True
WINDOW_SEC = 5.0
REPORT_INTERVAL_MS = 200

GRU_OVERRIDE_THRD_CLOT = 0.80
GRU_OVERRIDE_THRD_WALL = 0.92

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# ────────────────────────────────────────────────
# ClotFeatureExtractor
# ────────────────────────────────────────────────
class ClotFeatureExtractor:
    def __init__(self, sample_rate=150, window_sec=5.0):
        self.fs = sample_rate
        self.window_size = int(sample_rate * window_sec)
        self.buffer = deque(maxlen=self.window_size)
        self.ema_fast = 0.0
        self.ema_slow = 0.0
        self.alpha_fast = 0.2
        self.alpha_slow = 0.01
        self.zero_idx = [3, 6, 7, 25, 28, 29, 30, 31, 32, 33, 34, 35]

    def reset(self):
        self.buffer.clear()
        self.ema_fast = self.ema_slow = 0.0

    def update(self, r):
        self.buffer.append(float(r))
        if len(self.buffer) == 1:
            self.ema_fast = self.ema_slow = self.buffer[0]
        else:
            self.ema_fast = self.alpha_fast * self.buffer[-1] + (1 - self.alpha_fast) * self.ema_fast
            self.ema_slow = self.alpha_slow * self.buffer[-1] + (1 - self.alpha_slow) * self.ema_slow

    def compute_features(self):
        if len(self.buffer) < 100:
            return np.zeros(40, dtype=np.float32)

        data = np.array(self.buffer, dtype=np.float32)
        n = len(data)
        f = np.zeros(40, dtype=np.float32)
        i = 0

        # Basic stats
        f[i:i+10] = [data.mean(), data.std(), data.var(), data.min(), data.max(),
                     np.ptp(data), np.median(data),
                     np.std(data[-500:]) if n >= 500 else 0,
                     np.var(data[-500:]) if n >= 500 else 0,
                     np.mean(np.abs(np.diff(data[-500:]))) if n >= 500 else 0]
        i += 10

        # Slopes
        for secs in [1,2,3,4,5,6]:
            ns = min(int(secs * self.fs), n)
            if ns >= 2:
                slope = stats.linregress(np.arange(ns), data[-ns:]).slope
                f[i] = slope if np.isfinite(slope) else 0.0
            i += 1

        # Derivative
        deriv = np.diff(data)
        if len(deriv) > 10:
            f[i:i+6] = [deriv.mean(), deriv.std(), deriv.var(),
                        np.mean(np.abs(deriv)),
                        stats.skew(deriv) if len(deriv)>=3 else 0,
                        stats.kurtosis(deriv) if len(deriv)>=4 else 0]
        i += 6

        # EMA
        f[i:i+6] = [self.ema_fast, self.ema_slow, self.ema_fast-self.ema_slow,
                    0.0, self.ema_fast/(self.ema_slow+1e-6),
                    np.abs(self.ema_fast-self.ema_slow)]
        i += 6

        # Detrended
        kernel = 450
        if n >= kernel:
            trend = np.convolve(data, np.ones(kernel)/kernel, 'valid')
            detr = data[-len(trend):] - trend
            r600 = detr[-min(600,len(detr)):]
            f[i:i+8] = [np.std(r600), np.std(r600[:300]), np.mean(np.abs(r600)), 0.0,
                        np.std(np.diff(data[-500:])) if n>=500 else 0,
                        np.mean(np.abs(np.diff(data[-500:]))) if n>=500 else 0,
                        stats.skew(r600) if len(r600)>=3 else 0,
                        stats.kurtosis(r600) if len(r600)>=4 else 0]
        i += 8

        # Percentiles
        f[i:i+4] = [np.percentile(data,90)-data.mean(),
                    np.percentile(data,75)-np.percentile(data,25),
                    np.percentile(data,95)-np.percentile(data,5),
                    np.sum(data > np.percentile(data,95))/len(data)]

        return f.copy()


# ────────────────────────────────────────────────
# Dynamic dimension & paths
# ────────────────────────────────────────────────
active_dim = 40 - len(ClotFeatureExtractor().zero_idx) if REDUCE_DIM else 40
dim_str = f"red{active_dim}" if REDUCE_DIM else "40"

SCALER_PATH = PROJECT_ROOT / "src" / "data" / f"clot_feature_scaler_5s_seq{SEQ_LEN}_{dim_str}.pkl"
MODEL_PATH = PROJECT_ROOT / "src" / "training" / "clot_gru_trained.pt"
TEST_DATA_DIR = PROJECT_ROOT / "test_data"
OUTPUT_FOLDER = PROJECT_ROOT / "inference_deploy" / "Results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────────────────────────
# ClotGRU Model
# ────────────────────────────────────────────────
class ClotGRU(nn.Module):
    def __init__(self, input_size=None, hidden_size=32, output_size=3):
        super().__init__()
        if input_size is None:
            input_size = active_dim                     # Use dynamic size

        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

        nn.init.orthogonal_(self.gru.weight_ih_l0)
        nn.init.orthogonal_(self.gru.weight_hh_l0)
        nn.init.zeros_(self.gru.bias_ih_l0)
        nn.init.zeros_(self.gru.bias_hh_l0)

        self.fc1 = nn.Linear(hidden_size, 24)
        self.fc2 = nn.Linear(24, output_size)

        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x, hidden=None):
        # x shape: (batch, SEQ_LEN, input_size)
        out, hidden = self.gru(x, hidden)
        out = out[:, -1]
        out = torch.relu(self.fc1(out))
        logits = self.fc2(out)
        return logits, hidden


# ────────────────────────────────────────────────
# LiveClotDetector — FIXED
# ────────────────────────────────────────────────
class LiveClotDetector:
    def __init__(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
        self.scaler = joblib.load(scaler_path)
        self.model = ClotGRU().to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()

        self.hidden = None
        self.posterior = np.array([0.95, 0.025, 0.025], dtype=np.float32)
        
        # NEW: Feature history buffer for SEQ_LEN
        self.feat_history = deque(maxlen=SEQ_LEN)   # will store scaled active features

    @torch.no_grad()
    def predict(self, raw_feats_40, da_label=None):
        # Drop unused features before scaling
        if REDUCE_DIM:
            zero_idx = ClotFeatureExtractor().zero_idx
            active_idx = [i for i in range(40) if i not in zero_idx]
            feats_active = raw_feats_40[active_idx]
        else:
            feats_active = raw_feats_40

        scaled = self.scaler.transform(feats_active.reshape(1, -1))[0]

        # Add current scaled features to history buffer
        self.feat_history.append(scaled)

        # If history is not full yet, pad with the first available feature
        if len(self.feat_history) < SEQ_LEN:
            pad = list(self.feat_history)[0] if self.feat_history else scaled
            seq_list = [pad] * (SEQ_LEN - len(self.feat_history)) + list(self.feat_history)
        else:
            seq_list = list(self.feat_history)

        seq = np.array(seq_list, dtype=np.float32)   # (SEQ_LEN, feat_size)
        x = torch.from_numpy(seq).float().unsqueeze(0).to(DEVICE)  # (1, SEQ_LEN, feat_size)

        logits, self.hidden = self.model(x, self.hidden)
        if self.hidden is not None:
            self.hidden = self.hidden.detach()

        probs = torch.softmax(logits, 1).squeeze(0).cpu().numpy()
        
        prior_idx = np.argmax(self.posterior)
        
        if da_label is not None:
            if da_label == 0:  # DA says blood → force blood and reset everything
                self.posterior = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                self.hidden = None
                self.feat_history.clear()          # ← Important: reset history
                return self.posterior.copy()
            
            elif da_label in (1, 2):  # DA says clot or wall
                da_probs = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                da_probs[da_label] = 0.92
                da_probs[0] = 0.04
                da_probs[3 - da_label] = 0.04
                
                gru_top_idx = np.argmax(probs)
                if gru_top_idx != da_label:
                    thrd = GRU_OVERRIDE_THRD_CLOT if da_label == 1 else GRU_OVERRIDE_THRD_WALL
                    if probs[gru_top_idx] <= thrd:
                        probs = da_probs

        if prior_idx == 0:
            alpha_history, alpha_new = 0.78, 0.22
        else:
            if np.argmax(probs) == 0:
                alpha_history, alpha_new = 0.35, 0.65
            else:
                alpha_history, alpha_new = 0.96, 0.04
        
        self.posterior = alpha_history * self.posterior + alpha_new * probs       
        
        if da_label in (1, 2):
            final_idx = np.argmax(self.posterior)
            if final_idx != da_label:
                gru_top_idx = np.argmax(probs)
                thrd = GRU_OVERRIDE_THRD_CLOT if da_label == 1 else GRU_OVERRIDE_THRD_WALL
                if probs[gru_top_idx] < thrd:
                    da_probs = np.array([0.04, 0.04, 0.04], dtype=np.float32)
                    da_probs[da_label] = 0.92
                    self.posterior = da_probs

        return self.posterior.copy()

# ────────────────────────────────────────────────
#  Main Processing
# ────────────────────────────────────────────────

def process_file(filepath: Path,
                 all_gt_labels: list, 
                 all_da_labels: list, 
                 all_ml_preds: list, 
                 all_override_times: list):
    
    study_name = filepath.stem
    print(f"\nProcessing: {study_name}")

    df = pd.read_parquet(filepath)
    time_ms = df['timeInMS'].values          # ← exact column from your old code
    resistance = df['magRLoadAdjusted'].values.astype(np.float32)
    gt_labels = df.get('label', None)
    da_labels = df.get('da_label', None) if 'da_label' in df.columns else None

    extractor = ClotFeatureExtractor()
    detector = LiveClotDetector()

    results = []
    last_report = -REPORT_INTERVAL_MS

    for i, (t, r) in enumerate(zip(time_ms, resistance)):
        extractor.update(float(r))

        if t - last_report >= REPORT_INTERVAL_MS:
            feats_40 = extractor.compute_features()
            da_now = da_labels[i] if da_labels is not None else None
            post = detector.predict(feats_40, da_now)
            status = np.argmax(post)

            results.append({
                'time': t/1000.0,
                'prediction': status,
                'resistance': float(r),
                'Nprob': float(post[0]),
                'Cprob': float(post[1]),
                'Wprob': float(post[2])
            })
            last_report = t

    results_df = pd.DataFrame(results)

    # Save detection_results parquet (and csv)
    results_df.to_parquet(OUTPUT_FOLDER / f"{study_name}_detection_results.parquet", index=False)
    results_df.to_csv(OUTPUT_FOLDER / f"{study_name}_detection_results.csv", index=False)
    print(f"  Saved detection_results.parquet")

    # ── Probability plot (exact style from your image) ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Top: Detected Labels scatter
    colors = {0:'black', 1:'red', 2:'blue'}
    for lbl, name in [(0,'blood'),(1,'clot'),(2,'wall')]:
        mask = results_df['prediction'] == lbl
        ax1.scatter(results_df['time'][mask], results_df['resistance'][mask],
                    c=colors[lbl], s=4, label=name, alpha=0.8)
    ax1.set_ylabel('Resistance (Ω)')
    ax1.set_title(f'{study_name} — Detected Labels')
    ax1.grid(True, alpha=0.3)

    # Bottom: Only P(clot) & P(wall) with grey blood dominant shading
    ax2.plot(results_df['time'], results_df['Cprob'], color='red',   label='P(clot)', linewidth=1.8)
    ax2.plot(results_df['time'], results_df['Wprob'], color='blue',  label='P(wall)', linewidth=1.8)

    blood_dom = (results_df['Nprob'] > results_df['Cprob']) & (results_df['Nprob'] > results_df['Wprob'])
    ax2.fill_between(results_df['time'], 0, 1, where=blood_dom, color='gray', alpha=0.12, label='Blood dominant')

    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Probability')
    ax2.set_ylim(0, 1)
    ax2.set_title(f'{study_name} — Clot & Wall Detection Probabilities')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / f"{study_name}_detected_vs_clot_wall_probs.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved probability plot (exact style)")

    # ── Three-panel plot (exact style from your image) ──
    if gt_labels is not None and da_labels is not None:
        gt = gt_labels.values.astype(int)
        da = da_labels.values.astype(int)
        full_times = time_ms / 1000.0

        interp_ml = np.interp(full_times, results_df['time'], results_df['prediction'])
        interp_ml = np.round(interp_ml).astype(int)

        fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True, sharey=True,
                                 gridspec_kw={'height_ratios': [1,1,1]})

        colors = {0:'black', 1:'red', 2:'blue'}
        lbl_names = {0:'blood', 1:'clot', 2:'wall'}

        # Top: ML sparse with grey shading
        ax = axes[0]
        for lbl in [0,1,2]:
            mask = results_df['prediction'] == lbl
            ax.scatter(results_df['time'][mask], results_df['resistance'][mask],
                       c=colors[lbl], s=5, label=lbl_names[lbl], alpha=0.85)

        # Grey ML ≠ DA shading
        ml_da = np.interp(results_df['time'], full_times, da)
        ml_da = np.round(ml_da).astype(int)
        diff = (results_df['prediction'].values != ml_da)
        diff_diff = np.diff(diff.astype(int))
        starts = np.where(diff_diff == 1)[0] + 1
        ends = np.where(diff_diff == -1)[0] + 1
        if diff[0]: starts = np.insert(starts, 0, 0)
        if diff[-1]: ends = np.append(ends, len(diff))

        for s, e in zip(starts, ends):
            ax.axvspan(results_df['time'].iloc[s], results_df['time'].iloc[e-1],
                       facecolor='#e8e8e8', alpha=0.55, label='ML ≠ DA' if s==starts[0] else None)

        ax.set_title(f'{study_name} — ML Predictions (200 ms reporting)')
        ax.set_ylabel('Resistance (Ω)')
        ax.grid(True, alpha=0.3)

        # Middle & Bottom (DA & GT)
        for ax_idx, (title, data) in enumerate([("DA Labels (full 150 Hz)", da), ("Ground Truth Labels (full 150 Hz)", gt)]):
            ax = axes[ax_idx+1]
            for lbl in [0,1,2]:
                mask = data == lbl
                ax.scatter(full_times[mask], resistance[mask], c=colors[lbl], s=2, label=lbl_names[lbl], alpha=0.7)
            ax.set_title(f'{study_name} — {title}')
            ax.set_ylabel('Resistance (Ω)')
            ax.grid(True, alpha=0.3)
            if ax_idx == 1:
                ax.set_xlabel('Time (seconds)')

        plt.tight_layout(h_pad=0.8)
        plt.savefig(OUTPUT_FOLDER / f"{study_name}_ml_da_gt_three_panel.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved three-panel plot (exact style)")

        # Metrics
        print(f"\n{study_name} metrics:")
        print(f"DA  Acc: {accuracy_score(gt, da):.4f}  F1: {f1_score(gt, da, average='macro'):.4f} "
              f"Prec: {precision_score(gt, da, average='macro'):.4f}  Rec: {recall_score(gt, da, average='macro'):.4f}")
        print(f"ML  Acc: {accuracy_score(gt, interp_ml):.4f}  F1: {f1_score(gt, interp_ml, average='macro'):.4f} "
              f"Prec: {precision_score(gt, interp_ml, average='macro'):.4f}  Rec: {recall_score(gt, interp_ml, average='macro'):.4f}")
        print(f"Improvement: Acc {accuracy_score(gt, interp_ml)-accuracy_score(gt, da):+.4f}   "
              f"F1 {f1_score(gt, interp_ml, average='macro')-f1_score(gt, da, average='macro'):+.4f}")
        
        # Collect for global summary (only if ground truth & DA exist)
        if gt_labels is not None and da_labels is not None:
            # Interpolate ML predictions to full sampling points
    
            # Append to global lists
            all_gt_labels.extend(gt)
            all_da_labels.extend(da)
            all_ml_preds.extend(interp_ml)

        # Optional: count overrides (ML != DA)
        overrides = np.where(interp_ml != da)[0]
        all_override_times.extend(full_times[overrides])  # or just count len(overrides)
        
    print(f"Finished {study_name}\n")


# ────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────

def main():
    files = sorted(TEST_DATA_DIR.glob("*_labeled_segment.parquet"))
    print(f"Found {len(files)} labeled_segment files.\n")
    
    # ── Global aggregation lists ──
    all_gt_labels     = []
    all_da_labels     = []
    all_ml_preds      = []
    all_override_times = []  # optional — only if you track override moments

    for f in files:
        process_file(f,
                     all_gt_labels=all_gt_labels,
                     all_da_labels=all_da_labels,
                     all_ml_preds=all_ml_preds,
                     all_override_times=all_override_times)
        
    # ────────────────────────────────────────────────
    # Global summary – after ALL files processed
    # ────────────────────────────────────────────────
    if all_gt_labels:  # only if we have any ground truth data
        print("\n" + "="*70)
        print("GLOBAL SUMMARY ACROSS ALL STUDIES")
        print("="*70)

        acc_da  = accuracy_score(all_gt_labels, all_da_labels)
        f1_da   = f1_score(all_gt_labels, all_da_labels, average='macro', zero_division=0)
        prec_da = precision_score(all_gt_labels, all_da_labels, average='macro', zero_division=0)
        rec_da  = recall_score(all_gt_labels, all_da_labels, average='macro', zero_division=0)

        acc_ml  = accuracy_score(all_gt_labels, all_ml_preds)
        f1_ml   = f1_score(all_gt_labels, all_ml_preds, average='macro', zero_division=0)
        prec_ml = precision_score(all_gt_labels, all_ml_preds, average='macro', zero_division=0)
        rec_ml  = recall_score(all_gt_labels, all_ml_preds, average='macro', zero_division=0)

        print(f"DA  Accuracy: {acc_da:.4f}    F1-macro: {f1_da:.4f}")
        print(f"ML  Accuracy: {acc_ml:.4f}    F1-macro: {f1_ml:.4f}")
        print(f"Improvement: Acc {acc_ml - acc_da:+.4f}   F1 {f1_ml - f1_da:+.4f}")
        print("")
        print(f"DA  Precision: {prec_da:.4f}    Recall: {rec_da:.4f}")
        print(f"ML  Precision: {prec_ml:.4f}    Recall: {rec_ml:.4f}")
        print(f"Improvement: Precision {prec_ml - prec_da:+.4f}   Recall {rec_ml - rec_da:+.4f}")

        # total_overrides = len(all_override_times)
        # print(f"\nTotal ML overrides of DA: {total_overrides} points")    

    print("All files processed.")


if __name__ == "__main__":
    main()