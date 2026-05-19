"""
Inference pipeline for the 1D U-Net auto-labeler.

Processes full-length files in overlapping chunks, stitches predictions
with overlap averaging, and applies post-processing.

Usage:
    python -m auto_labeler.predict --input new_file.parquet --checkpoint auto_labeler/checkpoints/best_model.pt
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .dataset import CHUNK_SIZE, NUM_CHANNELS, NUM_CLASSES, build_multichannel
from .config import MIN_EVENT_DURATION_SEC, SAMPLING_RATE_HZ
from .model import UNet1D


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    """Load trained model from checkpoint. Returns (model, multichannel_flag)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt["args"]

    multichannel = args.get("multichannel", False)
    in_channels = NUM_CHANNELS if multichannel else 1

    model = UNet1D(
        in_channels=in_channels,
        num_classes=NUM_CLASSES,
        base_filters=args.get("base_filters", 32),
        depth=args.get("depth", 5),
        kernel_size=args.get("kernel_size", 7),
        dropout=args.get("dropout", 0.0),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, multichannel


@torch.no_grad()
def predict_file(
    model: UNet1D,
    features: np.ndarray,
    device: torch.device,
    chunk_size: int = CHUNK_SIZE,
    stride: int = None,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Run inference on a full-length signal.

    Uses overlapping chunks with softmax averaging for smooth predictions.

    Args:
        model: trained U-Net
        features: (num_channels, N) float32 array — multi-channel or (N,) for single
        device: torch device
        chunk_size: segment length
        stride: overlap stride (default: chunk_size // 2)
        batch_size: inference batch size

    Returns:
        labels: per-sample predicted labels (int64, same length as input)
    """
    if stride is None:
        stride = chunk_size // 2

    # Handle both single-channel (N,) and multi-channel (C, N)
    if features.ndim == 1:
        features = features[np.newaxis, :]  # (1, N)

    num_ch, n = features.shape

    # Accumulate softmax probabilities and counts for averaging
    prob_sum = np.zeros((NUM_CLASSES, n), dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)

    # Build chunks
    chunks = []
    starts = []
    pos = 0
    while pos + chunk_size <= n:
        chunks.append(features[:, pos : pos + chunk_size])  # (C, chunk_size)
        starts.append(pos)
        pos += stride

    # Handle tail
    if pos < n:
        tail = np.zeros((num_ch, chunk_size), dtype=np.float32)
        tail_len = n - pos
        tail[:, :tail_len] = features[:, pos:]
        tail[:, tail_len:] = features[:, -1:]
        chunks.append(tail)
        starts.append(pos)

    # Process in batches
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_starts = starts[i : i + batch_size]

        x = torch.tensor(np.array(batch_chunks), dtype=torch.float32)
        x = x.to(device)  # (B, C, L)

        logits = model(x)  # (B, C, L)
        probs = F.softmax(logits, dim=1).cpu().numpy()  # (B, C, L)

        for j, start in enumerate(batch_starts):
            end = min(start + chunk_size, n)
            valid_len = end - start
            prob_sum[:, start:end] += probs[j, :, :valid_len]
            counts[start:end] += 1.0

    # Average probabilities and take argmax
    counts = np.maximum(counts, 1.0)
    avg_probs = prob_sum / counts[np.newaxis, :]
    labels = avg_probs.argmax(axis=0).astype(np.int64)

    return labels


def postprocess_labels(
    labels: np.ndarray,
    min_duration_samples: int = 1000,  # ~6 seconds at 167 Hz
    sampling_rate_hz: float = 167.0,
) -> np.ndarray:
    """
    Post-processing: remove short spurious segments.

    Any contiguous segment of clot or wall shorter than min_duration_samples
    is replaced with the surrounding label (blood by default).
    """
    result = labels.copy()
    n = len(result)
    i = 0

    while i < n:
        current_label = result[i]
        # Find end of this segment
        j = i + 1
        while j < n and result[j] == current_label:
            j += 1

        segment_len = j - i

        # If non-blood segment is too short, replace with blood
        if current_label != 0 and segment_len < min_duration_samples:
            result[i:j] = 0

        i = j

    return result


def predict_parquet(
    input_path: str,
    checkpoint_path: str,
    output_path: str = None,
    min_duration_sec: float = MIN_EVENT_DURATION_SEC,
    device: str = None,
):
    """
    Full pipeline: load file → normalize → predict → postprocess → save.

    Args:
        input_path: path to input parquet (must have 'magRLoadAdjusted' column)
        checkpoint_path: path to trained model checkpoint
        output_path: where to save output parquet (default: input_labeled.parquet)
        min_duration_sec: minimum event duration for post-processing
        device: 'cuda' or 'cpu' (auto-detected if None)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print(f"Device: {device}")
    print(f"Loading model from: {checkpoint_path}")
    model, multichannel = load_model(checkpoint_path, device)

    print(f"Reading: {input_path}")
    df = pd.read_parquet(input_path)

    if "magRLoadAdjusted" not in df.columns:
        raise ValueError(f"Column 'magRLoadAdjusted' not found. Available: {list(df.columns)}")

    resistance = df["magRLoadAdjusted"].values.astype(np.float32)

    # Build features matching training
    if multichannel:
        features = build_multichannel(resistance)  # (5, N)
    else:
        mean = resistance.mean()
        std = resistance.std() + 1e-8
        features = ((resistance - mean) / std)[np.newaxis, :]  # (1, N)

    print(f"Predicting {len(resistance):,} samples ({features.shape[0]} channels)...")
    labels = predict_file(model, features, device)

    # Estimate sampling rate from time column if available
    sampling_rate = SAMPLING_RATE_HZ
    if "timeInMS" in df.columns:
        dt_ms = df["timeInMS"].diff().median()
        if dt_ms > 0:
            sampling_rate = 1000.0 / dt_ms

    # Post-process
    min_samples = int(min_duration_sec * sampling_rate)
    labels = postprocess_labels(labels, min_duration_samples=min_samples)

    # Build output
    df["predicted_label"] = labels
    print(f"Label distribution: blood={np.sum(labels==0):,}, clot={np.sum(labels==1):,}, wall={np.sum(labels==2):,}")

    if output_path is None:
        p = Path(input_path)
        output_path = str(p.parent / f"{p.stem}_auto_labeled{p.suffix}")

    df.to_parquet(output_path, index=False)
    print(f"Saved: {output_path}")
    return labels


def main():
    parser = argparse.ArgumentParser(description="Run auto-labeler inference on a parquet file")
    parser.add_argument("--input", type=str, required=True, help="Input parquet file")
    parser.add_argument("--checkpoint", type=str, default="auto_labeler/checkpoints/best_model.pt")
    parser.add_argument("--output", type=str, default=None, help="Output parquet path")
    parser.add_argument("--min_duration", type=float, default=MIN_EVENT_DURATION_SEC,
                        help="Min event duration (seconds)")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    args = parser.parse_args()

    predict_parquet(
        input_path=args.input,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        min_duration_sec=args.min_duration,
        device=args.device,
    )


if __name__ == "__main__":
    main()
