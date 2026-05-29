"""
Standalone inference script for the auto-labeler.

Runs prediction on parquet files without needing Colab or GPU.
All parameters are read from config.py.

Usage:
    python run_inference.py --input file.parquet
    python run_inference.py --input_dir folder_of_parquets/
    python run_inference.py --input file.parquet --checkpoint path/to/best_model.pt

Requirements:
    pip install torch numpy pandas pyarrow scipy
    (CPU-only torch works: pip install torch --index-url https://download.pytorch.org/whl/cpu)

Output:
    Each input parquet is saved with an added 'predicted_label' column (0=blood, 1=clot, 2=wall).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add parent directory to path so auto_labeler package can be imported
script_dir = Path(__file__).resolve().parent
if str(script_dir.parent) not in sys.path:
    sys.path.insert(0, str(script_dir.parent))

from auto_labeler import config as cfg
from auto_labeler.predict import load_model, predict_file, postprocess_labels
from auto_labeler.dataset import build_multichannel


def predict_parquet(filepath: Path, model, multichannel: bool, device: torch.device) -> dict:
    """Run prediction on a single parquet file. Returns summary dict."""
    df = pd.read_parquet(filepath)

    if "magRLoadAdjusted" not in df.columns:
        raise ValueError(f"Column 'magRLoadAdjusted' not found in {filepath.name}. "
                         f"Available: {list(df.columns)}")

    resistance = df["magRLoadAdjusted"].values.astype(np.float32)

    # Build features (normalization is per-file, no external scaler needed)
    if multichannel:
        features = build_multichannel(resistance)
    else:
        mean = resistance.mean()
        std = resistance.std() + 1e-8
        features = ((resistance - mean) / std)[np.newaxis, :]

    # Predict + post-process
    pred_labels = predict_file(model, features, device)
    pred_labels = postprocess_labels(pred_labels)

    # Add column and save
    df["predicted_label"] = pred_labels
    df.to_parquet(filepath, index=False)

    # Summary
    unique, counts = np.unique(pred_labels, return_counts=True)
    dist = {cfg.CLASS_NAMES[u]: int(c) for u, c in zip(unique, counts)}
    return {
        "file": filepath.name,
        "samples": len(pred_labels),
        "duration_sec": len(pred_labels) / cfg.SAMPLING_RATE_HZ,
        "distribution": dist,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run auto-labeler inference on parquet files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_inference.py --input study.parquet
  python run_inference.py --input_dir ./unlabeled_data/
  python run_inference.py --input study.parquet --checkpoint ./best_model.pt --device cpu
        """,
    )
    parser.add_argument("--input", type=str, help="Path to a single parquet file")
    parser.add_argument("--input_dir", type=str, help="Path to a directory of parquet files")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (default: auto_labeler/checkpoints/best_model.pt)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 'cpu' or 'cuda' (auto-detected if omitted)")
    args = parser.parse_args()

    if not args.input and not args.input_dir:
        parser.error("Provide --input or --input_dir")

    # Resolve checkpoint path
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        # Try common locations
        candidates = [
            script_dir / "checkpoints" / "best_model.pt",
            script_dir.parent / "auto_labeler" / "checkpoints" / "best_model.pt",
            Path("checkpoints") / "best_model.pt",
        ]
        checkpoint_path = None
        for c in candidates:
            if c.exists():
                checkpoint_path = c
                break
        if checkpoint_path is None:
            parser.error(f"Checkpoint not found. Tried: {[str(c) for c in candidates]}. "
                         f"Use --checkpoint to specify path.")

    # Collect input files
    parquets = []
    if args.input:
        p = Path(args.input)
        if not p.exists():
            parser.error(f"File not found: {p}")
        parquets.append(p)
    if args.input_dir:
        d = Path(args.input_dir)
        if not d.is_dir():
            parser.error(f"Directory not found: {d}")
        parquets.extend(sorted(d.glob("*.parquet")))

    if not parquets:
        print("No .parquet files found.")
        return

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    model, multichannel = load_model(str(checkpoint_path), device)
    print(f"Model loaded (channels={'multi(5)' if multichannel else 'single(1)'})")
    print(f"Post-processing: MIN_EVENT_DURATION = {cfg.MIN_EVENT_DURATION_SEC}s")
    print(f"\nProcessing {len(parquets)} file(s)...\n")

    # Run inference
    t0 = time.time()
    results = []
    for pf in parquets:
        try:
            summary = predict_parquet(pf, model, multichannel, device)
            results.append(summary)
            pct = {k: f"{v/summary['samples']*100:.1f}%" for k, v in summary["distribution"].items()}
            print(f"  {summary['file']:40s} | {summary['samples']:>8,} samples "
                  f"({summary['duration_sec']:.1f}s) | {pct}")
        except Exception as e:
            print(f"  ERROR {pf.name}: {e}")

    elapsed = time.time() - t0
    print(f"\nDone! {len(results)} files labeled in {elapsed:.1f}s")
    print(f"Labels saved in-place (added 'predicted_label' column).")


if __name__ == "__main__":
    main()
