"""
Evaluation tools for the auto-labeler.

Computes metrics and generates visual overlays comparing
predicted labels against ground truth.

Usage:
    python -m auto_labeler.evaluate --data_dir training_data --checkpoint auto_labeler/checkpoints/best_model.pt --studies SUMM0127 CENT0006
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from .dataset import CHUNK_SIZE, NUM_CLASSES, TEST_STUDIES, load_study
from .predict import load_model, postprocess_labels, predict_file


CLASS_NAMES = ["blood", "clot", "wall"]
CLASS_COLORS = ["blue", "red", "green"]


def evaluate_study(
    model, data_dir: Path, study_id: str, device: torch.device,
    min_duration_samples: int = 1000,
) -> dict:
    """Evaluate a single study: predict and compare to GT labels."""
    resistance, gt_labels = load_study(data_dir, study_id)

    # Normalize (same as training)
    mean = resistance.mean()
    std = resistance.std() + 1e-8
    resistance_norm = (resistance - mean) / std

    # Predict
    pred_labels = predict_file(model, resistance_norm, device)
    pred_labels = postprocess_labels(pred_labels, min_duration_samples=min_duration_samples)

    # Metrics
    n = len(gt_labels)
    acc = (pred_labels == gt_labels).mean()

    metrics = {"study": study_id, "n_samples": n, "accuracy": acc}

    for c in range(NUM_CLASSES):
        tp = ((pred_labels == c) & (gt_labels == c)).sum()
        fp = ((pred_labels == c) & (gt_labels != c)).sum()
        fn = ((pred_labels != c) & (gt_labels == c)).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        metrics[f"{CLASS_NAMES[c]}_f1"] = f1
        metrics[f"{CLASS_NAMES[c]}_precision"] = precision
        metrics[f"{CLASS_NAMES[c]}_recall"] = recall

    metrics["f1_macro"] = np.mean([metrics[f"{cn}_f1"] for cn in CLASS_NAMES])
    return metrics


def plot_overlay(
    data_dir: Path, study_id: str, model, device: torch.device,
    output_dir: Path, min_duration_samples: int = 1000,
    time_range: tuple = None,
):
    """Generate overlay plot: resistance + GT labels + predicted labels."""
    resistance, gt_labels = load_study(data_dir, study_id)

    mean = resistance.mean()
    std = resistance.std() + 1e-8
    resistance_norm = (resistance - mean) / std

    pred_labels = predict_file(model, resistance_norm, device)
    pred_labels = postprocess_labels(pred_labels, min_duration_samples=min_duration_samples)

    # Time axis (assume ~6ms per sample)
    dt = 0.006  # seconds
    time_sec = np.arange(len(resistance)) * dt

    if time_range:
        mask = (time_sec >= time_range[0]) & (time_sec <= time_range[1])
        idx = np.where(mask)[0]
    else:
        idx = np.arange(len(resistance))

    fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)

    # Top: resistance
    axes[0].plot(time_sec[idx], resistance[idx], "k-", linewidth=0.3, alpha=0.7)
    axes[0].set_ylabel("R (Ω)")
    axes[0].set_title(f"{study_id} — Resistance")

    # Middle: ground truth
    for c in range(NUM_CLASSES):
        mask_c = gt_labels[idx] == c
        if mask_c.any():
            axes[1].fill_between(
                time_sec[idx], 0, 1, where=mask_c,
                color=CLASS_COLORS[c], alpha=0.5, label=CLASS_NAMES[c],
            )
    axes[1].set_ylabel("GT Label")
    axes[1].legend(loc="upper right")
    axes[1].set_ylim(0, 1)

    # Bottom: predictions
    for c in range(NUM_CLASSES):
        mask_c = pred_labels[idx] == c
        if mask_c.any():
            axes[2].fill_between(
                time_sec[idx], 0, 1, where=mask_c,
                color=CLASS_COLORS[c], alpha=0.5, label=CLASS_NAMES[c],
            )
    axes[2].set_ylabel("Predicted")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(loc="upper right")
    axes[2].set_ylim(0, 1)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{study_id}_overlay.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate auto-labeler on studies with GT labels")
    parser.add_argument("--data_dir", type=str, default="training_data")
    parser.add_argument("--checkpoint", type=str, default="auto_labeler/checkpoints/best_model.pt")
    parser.add_argument("--output_dir", type=str, default="auto_labeler/results")
    parser.add_argument("--studies", nargs="*", default=None, help="Study IDs to evaluate (default: test set)")
    parser.add_argument("--min_duration", type=float, default=6.0)
    parser.add_argument("--plot", action="store_true", help="Generate overlay plots")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Determine which studies to evaluate
    study_ids = args.studies if args.studies else TEST_STUDIES

    # Check which studies actually exist in data_dir
    available = [sid for sid in study_ids if (data_dir / f"{sid}_labeled_segment.parquet").exists()]
    if not available:
        print(f"No studies found in {data_dir} for: {study_ids[:5]}...")
        return

    print(f"Evaluating {len(available)} studies...")

    # Sampling rate estimate
    sample_dt_ms = 6.0
    sampling_rate = 1000.0 / sample_dt_ms
    min_samples = int(args.min_duration * sampling_rate)

    results = []
    for sid in available:
        metrics = evaluate_study(model, data_dir, sid, device, min_duration_samples=min_samples)
        results.append(metrics)
        print(f"  {sid}: F1={metrics['f1_macro']:.4f}, Acc={metrics['accuracy']:.4f}")

        if args.plot:
            plot_overlay(data_dir, sid, model, device, output_dir / "plots", min_samples)

    # Summary
    df = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print(f"SUMMARY ({len(results)} studies)")
    print(f"{'='*60}")
    print(f"  Accuracy:   {df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f}")
    print(f"  F1 macro:   {df['f1_macro'].mean():.4f} ± {df['f1_macro'].std():.4f}")
    for cn in CLASS_NAMES:
        print(f"  F1 {cn:6s}: {df[f'{cn}_f1'].mean():.4f} ± {df[f'{cn}_f1'].std():.4f}")
    print(f"{'='*60}")

    # Save results CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "evaluation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved: {csv_path}")


if __name__ == "__main__":
    main()
