# gru_torch_V7.py
"""
Real-time clot detection — V7 (DA-as-feature)
Same as V6 but adds 2 DA fraction features to the GRU input:
  da_clot_frac: fraction of 5s window where DA says clot
  da_wall_frac: fraction of 5s window where DA says wall

Total input to GRU = 21 impedance features + 2 DA features = 23.
The DA override logic is REMOVED — DA information flows through the model only.
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
#from scipy import stats
#from scipy.signal import medfilt, find_peaks
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────
SEQ_LEN = 8
WINDOW_SEC = 5.0
REPORT_INTERVAL_MS = 200
SAMPLE_RATE = 150

# Temperature scaling for softmax
TEMPERATURE = 1.5

# ── Posterior EMA blending weights ──
EMA_BLOOD_PRIOR_HISTORY = 0.78
EMA_BLOOD_PRIOR_NEW     = 1 - EMA_BLOOD_PRIOR_HISTORY
EMA_EXIT_TO_BLOOD_HISTORY = 0.35
EMA_EXIT_TO_BLOOD_NEW     = 1 - EMA_EXIT_TO_BLOOD_HISTORY
EMA_SAME_CLASS_HISTORY  = 0.97
EMA_SAME_CLASS_NEW      = 1 - EMA_SAME_CLASS_HISTORY
EMA_CROSS_CLASS_HISTORY = 0.99
EMA_CROSS_CLASS_NEW     = 1 - EMA_CROSS_CLASS_HISTORY

# ── Initial posterior ──
INIT_BLOOD_PROB = 0.95
INIT_CLOT_PROB  = (1 - INIT_BLOOD_PROB) / 2
INIT_WALL_PROB  = (1 - INIT_BLOOD_PROB) / 2

# Feature config: 21 impedance + 2 DA = 23 total
FEATURE_SET = "clot_wall_focused_da"
IMPEDANCE_FEATURES = 21
DA_FEATURES = 2
TOTAL_INPUT_DIM = IMPEDANCE_FEATURES + DA_FEATURES  # 23

# Impedance feature indices (same as V6 clot_wall_focused)
IMPEDANCE_IDX = [39, 21, 4, 19, 41, 9, 5, 23, 0, 34, 28, 29, 3, 38, 17, 32, 42, 27, 1, 20, 40]
TOTAL_FEATURES = 64  # total impedance features computed by extractor

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Dynamic naming for scaler/model
_idx_hash = hash(tuple(IMPEDANCE_IDX) + ('da_frac',)) % 0xFFFF
dim_str = f"{FEATURE_SET}_{TOTAL_INPUT_DIM}_{_idx_hash:04x}"

SCALER_PATH = PROJECT_ROOT / "src" / "data" / f"clot_feature_scaler_5s_seq{SEQ_LEN}_{dim_str}.pkl"
MODEL_PATH = PROJECT_ROOT / "src" / "training" / "clot_gru_V7_trained.pt"
USE_DENOISED = False
SAVE_CSV_PARQUET = True
TEST_DATA_DIR = PROJECT_ROOT / ("test_data_denoised" if USE_DENOISED else "test_data")
OUTPUT_FOLDER = PROJECT_ROOT / "inference_deploy" / "Results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# For import by fit_scaler_V7 and train_gru_V7
active_idx = IMPEDANCE_IDX
active_dim = TOTAL_INPUT_DIM  # 23

# Window size in samples
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)  # 750


# ────────────────────────────────────────────────
# Import ClotFeatureExtractor from V6 (impedance features unchanged)
# ────────────────────────────────────────────────
from src.models.gru_torch_V6 import ClotFeatureExtractor


# ────────────────────────────────────────────────
# DA Fraction Buffer (streaming)
# ────────────────────────────────────────────────
class DAFractionBuffer:
    """Maintains a sliding window of DA labels and computes clot/wall fractions."""

    def __init__(self, window_size=WINDOW_SAMPLES):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def update(self, da_label):
        """Add a single DA label (0=blood, 1=clot, 2=wall)."""
        self.buffer.append(int(da_label) if da_label is not None else 0)

    def get_fractions(self):
        """Return (da_clot_frac, da_wall_frac) over the current window."""
        if len(self.buffer) == 0:
            return 0.0, 0.0
        buf = np.array(self.buffer)
        n = len(buf)
        da_clot_frac = float((buf == 1).sum()) / n
        da_wall_frac = float((buf == 2).sum()) / n
        return da_clot_frac, da_wall_frac

    def reset(self):
        self.buffer.clear()


def compute_da_fractions_from_array(da_window):
    """Batch mode: compute DA fractions from a window of DA labels."""
    n = len(da_window)
    if n == 0:
        return 0.0, 0.0
    da_clot_frac = float((da_window == 1).sum()) / n
    da_wall_frac = float((da_window == 2).sum()) / n
    return da_clot_frac, da_wall_frac


# ────────────────────────────────────────────────
# ClotGRU Model (same architecture, input_size=23)
# ────────────────────────────────────────────────
class ClotGRU(nn.Module):
    def __init__(self, input_size=None, hidden_size=64, output_size=3):
        super().__init__()
        if input_size is None:
            input_size = TOTAL_INPUT_DIM  # 23

        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

        nn.init.orthogonal_(self.gru.weight_ih_l0)
        nn.init.orthogonal_(self.gru.weight_hh_l0)
        nn.init.zeros_(self.gru.bias_ih_l0)
        nn.init.zeros_(self.gru.bias_hh_l0)

        self.fc1 = nn.Linear(hidden_size, 16)
        self.fc2 = nn.Linear(16, output_size)

        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x, hidden=None):
        out, hidden = self.gru(x, hidden)
        out = out[:, -1]
        out = torch.relu(self.fc1(out))
        logits = self.fc2(out)
        return logits, hidden


# ────────────────────────────────────────────────
# LiveClotDetector (NO DA override — DA is in the features)
# ────────────────────────────────────────────────
class LiveClotDetector:
    def __init__(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
        self.scaler = joblib.load(scaler_path)
        self.model = ClotGRU().to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()

        self.hidden = None
        self.posterior = np.array([INIT_BLOOD_PROB, INIT_CLOT_PROB, INIT_WALL_PROB],
                                  dtype=np.float32)
        self.raw_probs = self.posterior.copy()
        self.feat_history = deque(maxlen=SEQ_LEN)

    @torch.no_grad()
    def predict(self, active_feats, da_label=None):
        """
        Run one prediction step. active_feats is 23-dim (21 impedance + 2 DA fracs).
        Returns a 3-element posterior [P(blood), P(clot), P(wall)].

        DA blood reset is preserved as a safety net.
        Clot/wall DA override is removed — the model sees DA as input features.
        """
        # ── DA blood hard reset (safety net) ──
        if da_label is not None and da_label == 0:
            self.posterior = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            self.hidden = None
            self.feat_history.clear()
            return self.posterior.copy()

        # ── Step 1: Scale features & run GRU ──
        scaled = self.scaler.transform(active_feats.reshape(1, -1))[0]
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

        # ── Step 2: EMA blending ──
        if prior_idx == 0:
            alpha_history = EMA_BLOOD_PRIOR_HISTORY
            alpha_new     = EMA_BLOOD_PRIOR_NEW
        else:
            new_idx = np.argmax(probs)
            if new_idx == 0:
                alpha_history = EMA_EXIT_TO_BLOOD_HISTORY
                alpha_new     = EMA_EXIT_TO_BLOOD_NEW
            elif new_idx == prior_idx:
                alpha_history = EMA_SAME_CLASS_HISTORY
                alpha_new     = EMA_SAME_CLASS_NEW
            else:
                alpha_history = EMA_CROSS_CLASS_HISTORY
                alpha_new     = EMA_CROSS_CLASS_NEW

        self.posterior = alpha_history * self.posterior + alpha_new * probs
        return self.posterior.copy()


# ────────────────────────────────────────────────
#  Main Processing
# ────────────────────────────────────────────────

def process_file(filepath: Path,
                 all_gt_labels: list,
                 all_da_labels: list,
                 all_ml_preds: list,
                 all_override_times: list,
                 save_csv_parquet: bool = False):

    study_name = filepath.stem
    print(f"\nProcessing: {study_name}")

    df = pd.read_parquet(filepath)
    time_ms = df['timeInMS'].values
    resistance = df['magRLoadAdjusted'].values.astype(np.float32)
    gt_labels = df.get('label', None)
    da_labels = df.get('da_label', None) if 'da_label' in df.columns else None

    extractor = ClotFeatureExtractor(active_features=IMPEDANCE_IDX)
    da_buffer = DAFractionBuffer(window_size=WINDOW_SAMPLES)
    detector = LiveClotDetector()

    results = []
    last_report = -REPORT_INTERVAL_MS

    for i, (t, r) in enumerate(zip(time_ms, resistance)):
        extractor.update(float(r))
        da_now = int(da_labels[i]) if da_labels is not None else 0
        da_buffer.update(da_now)

        if t - last_report >= REPORT_INTERVAL_MS:
            # Get 21 impedance features
            impedance_feats = extractor.compute_features()

            # Get 2 DA fraction features
            da_clot_frac, da_wall_frac = da_buffer.get_fractions()

            # Concatenate: 23-dim feature vector
            full_feats = np.concatenate([impedance_feats,
                                         np.array([da_clot_frac, da_wall_frac], dtype=np.float32)])

            post = detector.predict(full_feats, da_label=da_now)
            raw = detector.raw_probs
            status = np.argmax(post)
            entropy = -np.sum(post * np.log(post + 1e-12))

            results.append({
                'time': t/1000.0,
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
    if save_csv_parquet:
        results_df.to_parquet(OUTPUT_FOLDER / f"{study_name}_detection_results.parquet", index=False)
        results_df.to_csv(OUTPUT_FOLDER / f"{study_name}_detection_results.csv", index=False)
        print(f"  Saved detection_results.parquet and .csv")

    # ── Probability plot (3 panels) ──
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14), sharex=True)

    colors = {0:'black', 1:'red', 2:'blue'}
    for lbl, name in [(0,'blood'),(1,'clot'),(2,'wall')]:
        mask = results_df['prediction'] == lbl
        ax1.scatter(results_df['time'][mask], results_df['resistance'][mask],
                    c=colors[lbl], s=4, label=name, alpha=0.8)
    ax1.set_ylabel('Resistance (Ω)')
    ax1.set_title(f'{study_name} — Detected Labels (V7: DA-as-feature)')
    ax1.grid(True, alpha=0.3)

    ax2.plot(results_df['time'], results_df['rawC'], color='red',   label='raw P(clot)', linewidth=1.2, alpha=0.8)
    ax2.plot(results_df['time'], results_df['rawW'], color='blue',  label='raw P(wall)', linewidth=1.2, alpha=0.8)
    ax2.set_ylabel('Probability')
    ax2.set_ylim(0, 1)
    ax2.set_title(f'{study_name} — Raw GRU Probabilities (T={TEMPERATURE})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.plot(results_df['time'], results_df['Cprob'], color='red',   label='P(clot)', linewidth=1.8)
    ax3.plot(results_df['time'], results_df['Wprob'], color='blue',  label='P(wall)', linewidth=1.8)
    blood_dom = (results_df['Nprob'] > results_df['Cprob']) & (results_df['Nprob'] > results_df['Wprob'])
    ax3.fill_between(results_df['time'], 0, 1, where=blood_dom, color='gray', alpha=0.12, label='Blood dominant')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Probability')
    ax3.set_ylim(0, 1)
    ax3.set_title(f'{study_name} — Smoothed Posterior')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / f"{study_name}_detected_vs_clot_wall_probs.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved probability plot")

    # ── Three-panel plot ──
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

        fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True, sharey=True,
                                 gridspec_kw={'height_ratios': [1,1,1]})

        lbl_names = {0:'blood', 1:'clot', 2:'wall'}

        ax = axes[0]
        for lbl in [0,1,2]:
            mask = results_df['prediction'] == lbl
            ax.scatter(results_df['time'][mask], results_df['resistance'][mask],
                       c=colors[lbl], s=5, label=lbl_names[lbl], alpha=0.85)

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

        ax.set_title(f'{study_name} — ML Predictions (V7, 200 ms)')
        ax.set_ylabel('Resistance (Ω)')
        ax.grid(True, alpha=0.3)

        for ax_idx, (title, data) in enumerate([("DA Labels (full 150 Hz)", da), ("Ground Truth Labels (full 150 Hz)", gt)]):
            ax = axes[ax_idx+1]
            unlabeled_mask = data == -1
            if unlabeled_mask.any():
                ax.scatter(full_times[unlabeled_mask], resistance[unlabeled_mask],
                           c='black', s=2, label='unlabeled', alpha=0.4, zorder=1)
            for lbl in [0,1,2]:
                mask = data == lbl
                ax.scatter(full_times[mask], resistance[mask], c=colors[lbl], s=2,
                           label=lbl_names[lbl], alpha=0.7, zorder=2)
            ax.set_title(f'{study_name} — {title}')
            ax.set_ylabel('Resistance (Ω)')
            ax.grid(True, alpha=0.3)
            if ax_idx == 1:
                ax.set_xlabel('Time (seconds)')

        plt.tight_layout(h_pad=0.8)
        plt.savefig(OUTPUT_FOLDER / f"{study_name}_ml_da_gt_three_panel.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved three-panel plot")

        # Metrics
        print(f"\n{study_name} metrics:")
        print(f"DA  Acc: {accuracy_score(gt_valid, da_valid):.4f}  F1: {f1_score(gt_valid, da_valid, average='macro'):.4f} "
              f"Prec: {precision_score(gt_valid, da_valid, average='macro'):.4f}  Rec: {recall_score(gt_valid, da_valid, average='macro'):.4f}")
        print(f"ML  Acc: {accuracy_score(gt_valid, ml_valid):.4f}  F1: {f1_score(gt_valid, ml_valid, average='macro'):.4f} "
              f"Prec: {precision_score(gt_valid, ml_valid, average='macro'):.4f}  Rec: {recall_score(gt_valid, ml_valid, average='macro'):.4f}")
        print(f"Improvement: Acc {accuracy_score(gt_valid, ml_valid)-accuracy_score(gt_valid, da_valid):+.4f}   "
              f"F1 {f1_score(gt_valid, ml_valid, average='macro')-f1_score(gt_valid, da_valid, average='macro'):+.4f}")

        # Override analysis
        override_mask = (ml_valid != da_valid)
        n_overrides = override_mask.sum()
        if n_overrides > 0:
            correct_overrides = ((ml_valid[override_mask] == gt_valid[override_mask]).sum())
            harmful_overrides = ((da_valid[override_mask] == gt_valid[override_mask]).sum())
            override_prec = correct_overrides / n_overrides

            da_cw_errors = ((da_valid != gt_valid) & ((gt_valid == 1) | (gt_valid == 2))).sum()
            override_rec = correct_overrides / da_cw_errors if da_cw_errors > 0 else 0.0

            print(f"\n  Override analysis ({study_name}):")
            print(f"    Total overrides:   {n_overrides}")
            print(f"    Correct (ML right, DA wrong): {correct_overrides}")
            print(f"    Harmful (DA right, ML wrong): {harmful_overrides}")
            print(f"    Override Precision: {override_prec:.4f}")
            print(f"    Override Recall:    {override_rec:.4f}  (of {da_cw_errors} DA clot/wall errors)")
        else:
            print(f"\n  No overrides in {study_name}")

        if gt_labels is not None and da_labels is not None:
            all_gt_labels.extend(gt_valid)
            all_da_labels.extend(da_valid)
            all_ml_preds.extend(ml_valid)

        overrides = np.where(ml_valid != da_valid)[0]
        all_override_times.extend(full_times[valid][overrides])

    print(f"Finished {study_name}\n")


# ────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────

def main():
    glob_pattern = "*_labeled_segment_denoised.parquet" if USE_DENOISED else "*_labeled_segment.parquet"
    files = sorted(TEST_DATA_DIR.glob(glob_pattern))
    print(f"Found {len(files)} files in {TEST_DATA_DIR.name}/\n")
    print(f"V7 config: {IMPEDANCE_FEATURES} impedance + {DA_FEATURES} DA = {TOTAL_INPUT_DIM} features")
    print(f"Scaler: {SCALER_PATH.name}")
    print(f"Model:  {MODEL_PATH.name}")

    all_gt_labels     = []
    all_da_labels     = []
    all_ml_preds      = []
    all_override_times = []

    for f in files:
        process_file(f,
                     all_gt_labels=all_gt_labels,
                     all_da_labels=all_da_labels,
                     all_ml_preds=all_ml_preds,
                     all_override_times=all_override_times,
                     save_csv_parquet=SAVE_CSV_PARQUET)

    # ── Global summary ──
    if all_gt_labels:
        summary_lines = []

        summary_lines.append("=" * 70)
        summary_lines.append("GLOBAL SUMMARY ACROSS ALL STUDIES (V7: DA-as-feature)")
        summary_lines.append("=" * 70)

        acc_da  = accuracy_score(all_gt_labels, all_da_labels)
        f1_da   = f1_score(all_gt_labels, all_da_labels, average='macro', zero_division=0)
        prec_da = precision_score(all_gt_labels, all_da_labels, average='macro', zero_division=0)
        rec_da  = recall_score(all_gt_labels, all_da_labels, average='macro', zero_division=0)

        acc_ml  = accuracy_score(all_gt_labels, all_ml_preds)
        f1_ml   = f1_score(all_gt_labels, all_ml_preds, average='macro', zero_division=0)
        prec_ml = precision_score(all_gt_labels, all_ml_preds, average='macro', zero_division=0)
        rec_ml  = recall_score(all_gt_labels, all_ml_preds, average='macro', zero_division=0)

        summary_lines.append(f"DA  Accuracy: {acc_da:.4f}    F1-macro: {f1_da:.4f}")
        summary_lines.append(f"ML  Accuracy: {acc_ml:.4f}    F1-macro: {f1_ml:.4f}")
        summary_lines.append(f"Improvement: Acc {acc_ml - acc_da:+.4f}   F1 {f1_ml - f1_da:+.4f}")
        summary_lines.append("")
        summary_lines.append(f"DA  Precision: {prec_da:.4f}    Recall: {rec_da:.4f}")
        summary_lines.append(f"ML  Precision: {prec_ml:.4f}    Recall: {rec_ml:.4f}")
        summary_lines.append(f"Improvement: Precision {prec_ml - prec_da:+.4f}   Recall {rec_ml - rec_da:+.4f}")

        gt_arr = np.array(all_gt_labels)
        da_arr = np.array(all_da_labels)
        ml_arr = np.array(all_ml_preds)

        g_override_mask = (ml_arr != da_arr)
        g_n_overrides = g_override_mask.sum()

        summary_lines.append(f"\n{'─'*70}")
        summary_lines.append(f"GLOBAL OVERRIDE ANALYSIS")
        summary_lines.append(f"{'─'*70}")
        summary_lines.append(f"Total overrides across all studies: {g_n_overrides}")

        if g_n_overrides > 0:
            g_correct = (ml_arr[g_override_mask] == gt_arr[g_override_mask]).sum()
            g_harmful = (da_arr[g_override_mask] == gt_arr[g_override_mask]).sum()
            g_neither = g_n_overrides - g_correct - g_harmful
            g_override_prec = g_correct / g_n_overrides

            g_da_cw_errors = ((da_arr != gt_arr) & ((gt_arr == 1) | (gt_arr == 2))).sum()
            g_override_rec = g_correct / g_da_cw_errors if g_da_cw_errors > 0 else 0.0

            summary_lines.append(f"  Correct overrides (ML right, DA wrong): {g_correct}")
            summary_lines.append(f"  Harmful overrides (DA right, ML wrong): {g_harmful}")
            summary_lines.append(f"  Neither correct (both wrong differently): {g_neither}")
            summary_lines.append(f"")
            summary_lines.append(f"  Override Precision: {g_override_prec:.4f}  (target: >0.85)")
            summary_lines.append(f"  Override Recall:    {g_override_rec:.4f}  (of {g_da_cw_errors} DA clot/wall errors)")
            summary_lines.append(f"  Net benefit:        {g_correct - g_harmful:+d} samples")

        summary_text = "\n".join(summary_lines)
        print("\n" + summary_text)

        summary_path = OUTPUT_FOLDER / "global_summary.txt"
        summary_path.write_text(summary_text, encoding="utf-8")
        print(f"\nSaved global summary to {summary_path.name}")

    print("\nAll files processed.")


if __name__ == "__main__":
    main()
