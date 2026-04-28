# diagnose_clot_wall_confusion.py
"""
Diagnostic script: why does the GRU confidently predict clot in a flat wall segment?

Compares features between:
  - The misclassified region (GRU says clot, GT says wall)
  - Correctly classified wall regions in the same study
  - Correctly classified clot regions in the same study

Outputs:
  1. Feature comparison table (scaled values, z-scores vs wall/clot distributions)
  2. Per-feature "blame score" — how much each feature looks like clot vs wall
  3. Raw logit analysis at different temperatures
  4. Plots: feature radar chart + logit sensitivity
"""

import sys
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

from gru_torch_V6 import (
    ClotFeatureExtractor, ClotGRU, LiveClotDetector,
    active_idx, active_dim, SEQ_LEN, TEMPERATURE,
    SCALER_PATH, MODEL_PATH, DEVICE,
    PROJECT_ROOT, REPORT_INTERVAL_MS, FEATURE_SETS, FEATURE_SET
)

# ═══════════════════════════════════════════════
# CONFIG — adjust these for your case
# ═══════════════════════════════════════════════
STUDY_FILE = "STCLD001_labeled_segment.parquet"
# Time range (seconds) of the misclassified region from the plot
PROBLEM_START_SEC = 1600.0
PROBLEM_END_SEC   = 1800.0
# ═══════════════════════════════════════════════

TEST_DATA_DIR = PROJECT_ROOT / "test_data"

# Feature names for the clot_wall_focused set (21 features in order)
ALL_FEAT_NAMES = {
    0: "mean", 1: "std", 2: "var", 3: "min", 4: "max", 5: "range",
    6: "median", 7: "win_std", 8: "win_var", 9: "win_diff",
    10: "slope_1s", 11: "slope_2s", 12: "slope_3s", 13: "slope_4s",
    14: "slope_5s", 15: "slope_6s",
    16: "deriv_mean", 17: "deriv_std", 18: "deriv_var", 19: "deriv_mean_abs",
    20: "deriv_skew", 21: "deriv_kurtosis",
    22: "ema_fast", 23: "ema_slow", 24: "ema_diff", 25: "ema_zero",
    26: "ema_ratio", 27: "ema_abs_diff",
    28: "detr_std", 29: "detr_std300", 30: "detr_mean_abs", 31: "detr_zero",
    32: "detr_std_diff500", 33: "detr_mean_abs_d500", 34: "detr_skew", 35: "detr_kurtosis",
    36: "p90_mean", 37: "IQR", 38: "p95_p5", 39: "frac_above_p95",
    40: "hjorth_mob", 41: "hjorth_complex", 42: "mean_abs_2nd_deriv",
}


def get_active_feat_names():
    """Return ordered feature names for the active feature set."""
    return [ALL_FEAT_NAMES.get(i, f"f{i}") for i in active_idx]


def extract_features_for_region(resistance, time_ms, start_sec, end_sec, sample_rate=150):
    """Extract features at REPORT_INTERVAL_MS intervals for a time region."""
    extractor = ClotFeatureExtractor(sample_rate=sample_rate, active_features=active_idx)
    features = []
    times = []
    last_report = -REPORT_INTERVAL_MS

    for i, (t, r) in enumerate(zip(time_ms, resistance)):
        extractor.update(float(r))
        t_sec = t / 1000.0

        if t - last_report >= REPORT_INTERVAL_MS and t_sec >= start_sec - 10:
            feats = extractor.compute_features()
            if start_sec <= t_sec <= end_sec and np.any(feats != 0):
                features.append(feats.copy())
                times.append(t_sec)
            last_report = t

        if t_sec > end_sec + 5:
            break

    return np.array(features), np.array(times)


def extract_features_by_label(resistance, time_ms, gt_labels, target_label, sample_rate=150, max_samples=200):
    """Extract features from regions where GT == target_label."""
    extractor = ClotFeatureExtractor(sample_rate=sample_rate, active_features=active_idx)
    features = []
    last_report = -REPORT_INTERVAL_MS

    for i, (t, r) in enumerate(zip(time_ms, resistance)):
        extractor.update(float(r))

        if t - last_report >= REPORT_INTERVAL_MS:
            if gt_labels[i] == target_label:
                feats = extractor.compute_features()
                if np.any(feats != 0):
                    features.append(feats.copy())
            last_report = t

        if len(features) >= max_samples:
            break

    # Also sample from the back half
    if len(features) < max_samples:
        extractor2 = ClotFeatureExtractor(sample_rate=sample_rate, active_features=active_idx)
        mid = len(time_ms) // 2
        last_report = time_ms[mid] - REPORT_INTERVAL_MS
        for i in range(mid, len(time_ms)):
            t, r = time_ms[i], resistance[i]
            extractor2.update(float(r))
            if t - last_report >= REPORT_INTERVAL_MS:
                if gt_labels[i] == target_label:
                    feats = extractor2.compute_features()
                    if np.any(feats != 0):
                        features.append(feats.copy())
                last_report = t
            if len(features) >= max_samples:
                break

    return np.array(features) if features else np.zeros((0, active_dim))


def run_gru_on_features(features, scaler, model):
    """Run the GRU on a batch of feature vectors and return logits + probs at various temps."""
    model.eval()
    all_logits = []
    all_probs = {}

    for T in [1.0, 1.5, 2.0, 3.0, 5.0]:
        all_probs[T] = []

    with torch.no_grad():
        for feat in features:
            scaled = scaler.transform(feat.reshape(1, -1))[0]
            # Repeat to fill a sequence (simple diagnostic — ignores history effects)
            seq = np.tile(scaled, (SEQ_LEN, 1)).astype(np.float32)
            x = torch.from_numpy(seq).float().unsqueeze(0).to(DEVICE)
            logits, _ = model(x, None)
            logits_np = logits.squeeze(0).cpu().numpy()
            all_logits.append(logits_np)

            for T in all_probs:
                probs = torch.softmax(logits / T, 1).squeeze(0).cpu().numpy()
                all_probs[T].append(probs)

    return np.array(all_logits), {T: np.array(v) for T, v in all_probs.items()}


def main():
    filepath = TEST_DATA_DIR / STUDY_FILE
    if not filepath.exists():
        print(f"File not found: {filepath}")
        print(f"Available files:")
        for f in sorted(TEST_DATA_DIR.glob("*.parquet")):
            print(f"  {f.name}")
        sys.exit(1)

    print(f"Loading {STUDY_FILE}...")
    df = pd.read_parquet(filepath)
    time_ms = df['timeInMS'].values
    resistance = df['magRLoadAdjusted'].values.astype(np.float32)
    gt_labels = df['label'].values.astype(int) if 'label' in df.columns else None

    if gt_labels is None:
        print("No 'label' column in data — cannot compare to ground truth")
        sys.exit(1)

    scaler = joblib.load(SCALER_PATH)
    model = ClotGRU().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    feat_names = get_active_feat_names()

    # ── Extract features ──
    print(f"\nExtracting features for problem region: {PROBLEM_START_SEC}-{PROBLEM_END_SEC}s...")
    problem_feats, problem_times = extract_features_for_region(
        resistance, time_ms, PROBLEM_START_SEC, PROBLEM_END_SEC)
    print(f"  Got {len(problem_feats)} feature vectors")

    print(f"Extracting features for GT=wall (label=2) regions...")
    wall_feats = extract_features_by_label(resistance, time_ms, gt_labels, target_label=2)
    print(f"  Got {len(wall_feats)} wall feature vectors")

    print(f"Extracting features for GT=clot (label=1) regions...")
    clot_feats = extract_features_by_label(resistance, time_ms, gt_labels, target_label=1)
    print(f"  Got {len(clot_feats)} clot feature vectors")

    if len(problem_feats) == 0:
        print("No features extracted for problem region. Check PROBLEM_START_SEC/PROBLEM_END_SEC.")
        sys.exit(1)

    # ── Scale all features ──
    problem_scaled = scaler.transform(problem_feats)
    wall_scaled = scaler.transform(wall_feats) if len(wall_feats) > 0 else np.zeros((0, active_dim))
    clot_scaled = scaler.transform(clot_feats) if len(clot_feats) > 0 else np.zeros((0, active_dim))

    # ── Compute statistics ──
    prob_mean = problem_scaled.mean(axis=0)
    wall_mean = wall_scaled.mean(axis=0) if len(wall_scaled) > 0 else np.zeros(active_dim)
    wall_std  = wall_scaled.std(axis=0) + 1e-8 if len(wall_scaled) > 0 else np.ones(active_dim)
    clot_mean = clot_scaled.mean(axis=0) if len(clot_scaled) > 0 else np.zeros(active_dim)
    clot_std  = clot_scaled.std(axis=0) + 1e-8 if len(clot_scaled) > 0 else np.ones(active_dim)

    # Z-scores: how many SDs is the problem region from the wall/clot distributions?
    z_vs_wall = (prob_mean - wall_mean) / wall_std
    z_vs_clot = (prob_mean - clot_mean) / clot_std

    # Blame score: positive = looks more like clot, negative = looks more like wall
    # Using normalized distance: closer to clot mean = more blame
    dist_to_wall = np.abs(prob_mean - wall_mean)
    dist_to_clot = np.abs(prob_mean - clot_mean)
    blame = dist_to_wall - dist_to_clot  # positive = closer to clot (bad)

    # ── Print feature comparison table ──
    print("\n" + "=" * 110)
    print("FEATURE COMPARISON: Problem region vs Wall vs Clot (all values are SCALED)")
    print("=" * 110)
    print(f"{'Feature':<22} {'Problem':>9} {'Wall μ':>9} {'Clot μ':>9} "
          f"{'z(wall)':>9} {'z(clot)':>9} {'Blame':>9}  Verdict")
    print("-" * 110)

    blame_order = np.argsort(-blame)  # most clot-like first
    for i in blame_order:
        verdict = ""
        if blame[i] > 0.5:
            verdict = "⚠ LOOKS LIKE CLOT"
        elif blame[i] > 0.2:
            verdict = "~ leans clot"
        elif blame[i] < -0.5:
            verdict = "✓ looks like wall"

        print(f"  {feat_names[i]:<20} {prob_mean[i]:>9.3f} {wall_mean[i]:>9.3f} {clot_mean[i]:>9.3f} "
              f"{z_vs_wall[i]:>9.2f} {z_vs_clot[i]:>9.2f} {blame[i]:>9.3f}  {verdict}")

    # ── Top blame features ──
    print("\n" + "=" * 70)
    print("TOP 5 FEATURES MAKING THIS LOOK LIKE CLOT (highest blame):")
    print("=" * 70)
    for rank, i in enumerate(blame_order[:5], 1):
        print(f"  {rank}. {feat_names[i]:<22} blame={blame[i]:+.3f}  "
              f"problem={prob_mean[i]:.3f}  wall={wall_mean[i]:.3f}  clot={clot_mean[i]:.3f}")

    print("\nTOP 5 FEATURES THAT CORRECTLY LOOK LIKE WALL (lowest blame):")
    for rank, i in enumerate(blame_order[-5:][::-1], 1):
        print(f"  {rank}. {feat_names[i]:<22} blame={blame[i]:+.3f}  "
              f"problem={prob_mean[i]:.3f}  wall={wall_mean[i]:.3f}  clot={clot_mean[i]:.3f}")

    # ── GRU logit / temperature analysis ──
    print("\n" + "=" * 70)
    print("GRU LOGIT & TEMPERATURE ANALYSIS (problem region)")
    print("=" * 70)

    logits, probs_by_T = run_gru_on_features(problem_feats, scaler, model)
    mean_logits = logits.mean(axis=0)
    print(f"\n  Mean raw logits:  blood={mean_logits[0]:.3f}  clot={mean_logits[1]:.3f}  wall={mean_logits[2]:.3f}")
    print(f"  Logit gap (clot - wall): {mean_logits[1] - mean_logits[2]:.3f}")
    print(f"  Logit gap (clot - blood): {mean_logits[1] - mean_logits[0]:.3f}")

    print(f"\n  {'Temp':>6}  {'P(blood)':>10}  {'P(clot)':>10}  {'P(wall)':>10}  {'Winner':>8}")
    print(f"  {'-'*50}")
    for T in sorted(probs_by_T.keys()):
        mp = probs_by_T[T].mean(axis=0)
        winner = ['blood', 'clot', 'wall'][np.argmax(mp)]
        marker = " ← current" if abs(T - TEMPERATURE) < 0.01 else ""
        print(f"  {T:>6.1f}  {mp[0]:>10.4f}  {mp[1]:>10.4f}  {mp[2]:>10.4f}  {winner:>8}{marker}")

    # ── Also run on wall regions for comparison ──
    if len(wall_feats) > 0:
        print(f"\n  For comparison — correctly classified WALL regions:")
        wall_logits, wall_probs_T = run_gru_on_features(wall_feats[:50], scaler, model)
        wml = wall_logits.mean(axis=0)
        print(f"  Mean raw logits:  blood={wml[0]:.3f}  clot={wml[1]:.3f}  wall={wml[2]:.3f}")
        wmp = wall_probs_T[TEMPERATURE].mean(axis=0)
        print(f"  Mean probs (T={TEMPERATURE}): blood={wmp[0]:.4f}  clot={wmp[1]:.4f}  wall={wmp[2]:.4f}")

    # ── Raw (unscaled) feature comparison for interpretability ──
    print("\n" + "=" * 70)
    print("RAW (UNSCALED) FEATURE VALUES — for physical interpretation")
    print("=" * 70)
    prob_raw_mean = problem_feats.mean(axis=0)
    wall_raw_mean = wall_feats.mean(axis=0) if len(wall_feats) > 0 else np.zeros(active_dim)
    clot_raw_mean = clot_feats.mean(axis=0) if len(clot_feats) > 0 else np.zeros(active_dim)

    print(f"  {'Feature':<22} {'Problem':>12} {'Wall':>12} {'Clot':>12}")
    print(f"  {'-'*60}")
    for i in blame_order[:10]:
        print(f"  {feat_names[i]:<22} {prob_raw_mean[i]:>12.4f} {wall_raw_mean[i]:>12.4f} {clot_raw_mean[i]:>12.4f}")

    # ── Plot ──
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Blame bar chart
    ax = axes[0]
    colors = ['red' if b > 0.2 else 'blue' if b < -0.2 else 'gray' for b in blame[blame_order]]
    ax.barh(range(len(blame)), blame[blame_order], color=colors)
    ax.set_yticks(range(len(blame)))
    ax.set_yticklabels([feat_names[i] for i in blame_order], fontsize=8)
    ax.set_xlabel("Blame score (+ = looks like clot, − = looks like wall)")
    ax.set_title(f"Feature blame: {STUDY_FILE} @ {PROBLEM_START_SEC}-{PROBLEM_END_SEC}s")
    ax.axvline(0, color='black', linewidth=0.5)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    # Temperature sensitivity
    ax = axes[1]
    temps = sorted(probs_by_T.keys())
    blood_means = [probs_by_T[T].mean(axis=0)[0] for T in temps]
    clot_means  = [probs_by_T[T].mean(axis=0)[1] for T in temps]
    wall_means  = [probs_by_T[T].mean(axis=0)[2] for T in temps]
    ax.plot(temps, blood_means, 'k-o', label='P(blood)')
    ax.plot(temps, clot_means,  'r-o', label='P(clot)')
    ax.plot(temps, wall_means,  'b-o', label='P(wall)')
    ax.axvline(TEMPERATURE, color='gray', linestyle='--', label=f'current T={TEMPERATURE}')
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Mean probability")
    ax.set_title("Temperature sensitivity — problem region")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    out_path = Path(__file__).parent / "diagnose_clot_wall_confusion.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {out_path}")


if __name__ == "__main__":
    main()
