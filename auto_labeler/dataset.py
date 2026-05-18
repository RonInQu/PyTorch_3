"""
Dataset for 1D U-Net auto-labeler.

Reads labeled parquet files and produces fixed-length chunks
of raw resistance with per-sample labels for segmentation training.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Optional


# 85 training studies from production baseline manifest (2026-05-18)
TRAINING_STUDIES = [
    "00F628C9", "05EA15A5", "09419CF3", "0D9C36A0", "15A93526",
    "15AC6217", "16621B3E", "18A9A741", "1A8F0795", "24AFD80C",
    "26E955BA", "325A317A", "34268034", "376DCB0D", "3B90D74B",
    "3E146478", "42CF0AE3", "43140EA7", "453F37DC", "4633BDC0",
    "48663E05", "4B4BF4DB", "4E3747A0", "50ACAF6E", "530618CC",
    "58F78079", "5A31F836", "6E7EB56C", "71119917", "73CB9CA1",
    "743CBF58", "7873BF1D", "81FC0C79", "86FA6755", "8860D580",
    "8EE40C79", "903FE519", "9C63125D", "A225B105", "AFF18ECE",
    "B58B74D7", "B9E8EB7F", "BAPT0001", "CENT0006", "CENT0007",
    "CENT0009", "CENT0102", "CENT0161", "CENT0165", "CENT0176",
    "CENT0182", "CENT0231", "D25DD102", "D4793E80", "DBEF90C4",
    "EA7C0500", "ELCA0179", "F39B2DEA", "F60DF902", "FE454F2D",
    "HACK0140", "HUNT0120", "HUNT0130", "HUNT0134", "HUNT0136",
    "HUNT0150", "HUNT0159", "HUNT0177", "HUNT0178", "HUNT0198",
    "NASHUN01", "NASHUN02", "SOMI0153", "STCL0090", "STCLD001",
    "STCLD002", "SUMM0119", "SUMM0149", "SUMM0152", "SUMM0154",
    "SUMM0163", "SUMM0183", "UH000008", "UHMAX001", "UNIH0148",
]

TEST_STUDIES = [
    "39265C2B", "B9E5A9D2", "BADE209A", "BAPT0282", "BAPT0291",
    "C245C6AB", "CENT0237", "CENT0249", "CENT0277", "CLCL0232",
    "F4F385C8", "HUNT0275", "HUNT0288", "LINC0194", "RJWN0278",
    "RJWN0279", "SLOA0197", "SUMM0226", "SUMM0242", "SUMM0243",
    "WEKE0283",
]

NUM_CLASSES = 3
CHUNK_SIZE = 4096  # ~24.5 seconds at 167 Hz


def load_study(data_dir: Path, study_id: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a single study's resistance and labels."""
    path = data_dir / f"{study_id}_labeled_segment.parquet"
    df = pd.read_parquet(path, columns=["magRLoadAdjusted", "label"])
    resistance = df["magRLoadAdjusted"].values.astype(np.float32)
    labels = df["label"].values.astype(np.int64)
    return resistance, labels


def compute_class_weights(data_dir: Path, study_ids: List[str]) -> torch.Tensor:
    """Compute inverse-frequency class weights from training data."""
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for sid in study_ids:
        _, labels = load_study(data_dir, sid)
        for c in range(NUM_CLASSES):
            counts[c] += (labels == c).sum()
    # Inverse frequency, normalized so min weight = 1.0
    weights = counts.sum() / (NUM_CLASSES * counts.astype(np.float64))
    weights = weights / weights.min()
    return torch.tensor(weights, dtype=torch.float32)


class SegmentationDataset(Dataset):
    """
    Dataset that chunks studies into fixed-length segments for U-Net training.

    Each sample is:
      - x: (1, CHUNK_SIZE) float32 — normalized resistance
      - y: (CHUNK_SIZE,) int64 — per-sample labels (0/1/2)
    """

    def __init__(
        self,
        data_dir: str,
        study_ids: List[str],
        chunk_size: int = CHUNK_SIZE,
        stride: Optional[int] = None,
        normalize: bool = True,
        augment: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.stride = stride if stride is not None else chunk_size // 2
        self.normalize = normalize
        self.augment = augment

        # Preload all data and build chunk index
        self.chunks: List[Tuple[np.ndarray, np.ndarray]] = []
        self._load_all(study_ids)

    def _load_all(self, study_ids: List[str]):
        """Load studies and create overlapping chunks."""
        for sid in study_ids:
            resistance, labels = load_study(self.data_dir, sid)

            # Per-study normalization: z-score
            if self.normalize:
                mean = resistance.mean()
                std = resistance.std() + 1e-8
                resistance = (resistance - mean) / std

            n = len(resistance)
            start = 0
            while start + self.chunk_size <= n:
                r_chunk = resistance[start : start + self.chunk_size]
                l_chunk = labels[start : start + self.chunk_size]
                self.chunks.append((r_chunk, l_chunk))
                start += self.stride

            # Handle tail: pad last chunk if remaining > chunk_size // 4
            remaining = n - start
            if remaining > self.chunk_size // 4:
                # Pad with last value (resistance) and blood label (0)
                r_chunk = np.zeros(self.chunk_size, dtype=np.float32)
                l_chunk = np.full(self.chunk_size, 0, dtype=np.int64)
                r_chunk[:remaining] = resistance[start:]
                l_chunk[:remaining] = labels[start:]
                r_chunk[remaining:] = resistance[-1]
                self.chunks.append((r_chunk, l_chunk))

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        r_chunk, l_chunk = self.chunks[idx]

        if self.augment:
            r_chunk = self._augment(r_chunk.copy())

        x = torch.from_numpy(r_chunk).unsqueeze(0)  # (1, chunk_size)
        y = torch.from_numpy(l_chunk)  # (chunk_size,)
        return x, y

    def _augment(self, x: np.ndarray) -> np.ndarray:
        """Simple augmentations for training."""
        # Gaussian noise
        if np.random.rand() < 0.5:
            x += np.random.randn(len(x)).astype(np.float32) * 0.02

        # Amplitude scaling
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            x *= scale

        # DC offset shift
        if np.random.rand() < 0.3:
            x += np.random.uniform(-0.1, 0.1)

        return x


def create_datasets(
    data_dir: str,
    val_fraction: float = 0.15,
    chunk_size: int = CHUNK_SIZE,
    stride: Optional[int] = None,
    seed: int = 42,
) -> Tuple[SegmentationDataset, SegmentationDataset, List[str], List[str]]:
    """
    Create train and validation datasets with study-level split.

    Returns: (train_dataset, val_dataset, train_ids, val_ids)
    """
    rng = np.random.default_rng(seed)
    ids = TRAINING_STUDIES.copy()
    rng.shuffle(ids)

    n_val = max(1, int(len(ids) * val_fraction))
    val_ids = ids[:n_val]
    train_ids = ids[n_val:]

    print(f"Train: {len(train_ids)} studies, Val: {len(val_ids)} studies")

    train_ds = SegmentationDataset(
        data_dir, train_ids, chunk_size=chunk_size,
        stride=stride, augment=True,
    )
    val_ds = SegmentationDataset(
        data_dir, val_ids, chunk_size=chunk_size,
        stride=chunk_size,  # no overlap for validation
        augment=False,
    )

    print(f"Train chunks: {len(train_ds)}, Val chunks: {len(val_ds)}")
    return train_ds, val_ds, train_ids, val_ids
