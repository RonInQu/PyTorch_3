"""
Training script for the 1D U-Net auto-labeler.

ALL parameters are read from config.py — no CLI overrides.

Usage:
    python -m auto_labeler.train --data_dir training_data --output_dir checkpoints
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config as cfg
from .dataset import (
    TRAINING_STUDIES,
    compute_class_weights,
    create_datasets,
)
from .model import UNet1D, count_parameters


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced segmentation."""

    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(
            logits, targets, weight=self.alpha, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def compute_metrics(preds: torch.Tensor, labels: torch.Tensor, num_classes: int = cfg.NUM_CLASSES):
    """Compute per-class F1, overall accuracy, and IoU."""
    pred_flat = preds.argmax(dim=1).view(-1)
    label_flat = labels.view(-1)

    acc = (pred_flat == label_flat).float().mean().item()

    f1s = []
    ious = []
    for c in range(num_classes):
        tp = ((pred_flat == c) & (label_flat == c)).sum().float()
        fp = ((pred_flat == c) & (label_flat != c)).sum().float()
        fn = ((pred_flat != c) & (label_flat == c)).sum().float()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)

        f1s.append(f1.item())
        ious.append(iou.item())

    return {
        "accuracy": acc,
        "f1_macro": np.mean(f1s),
        "f1_per_class": f1s,
        "iou_macro": np.mean(ious),
        "iou_per_class": ious,
    }


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)  # (B, C, L)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.GRAD_CLIP_NORM)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)

    return total_loss / total_samples


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)
        all_preds.append(logits.cpu())
        all_labels.append(y.cpu())

    avg_loss = total_loss / total_samples
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(all_preds, all_labels)
    metrics["loss"] = avg_loss
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train 1D U-Net auto-labeler")
    parser.add_argument("--data_dir", type=str, default="training_data",
                        help="Path to training parquet files")
    parser.add_argument("--output_dir", type=str, default="auto_labeler/checkpoints",
                        help="Path to save checkpoints and manifest")
    args = parser.parse_args()

    # ─── All parameters from config.py ────────────────────────────────────
    multichannel = cfg.NUM_CHANNELS > 1
    in_channels = cfg.NUM_CHANNELS
    stride = cfg.TRAIN_STRIDE if cfg.TRAIN_STRIDE else cfg.CHUNK_SIZE // 2

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create datasets
    print("Loading data...")
    train_ds, val_ds, train_ids, val_ids = create_datasets(
        data_dir=args.data_dir,
        val_fraction=cfg.VAL_FRACTION,
        chunk_size=cfg.CHUNK_SIZE,
        stride=stride,
        multichannel=multichannel,
        seed=cfg.SEED,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )

    # Class weights
    print("Computing class weights...")
    class_weights = compute_class_weights(Path(args.data_dir), train_ids).to(device)
    print(f"Class weights: {class_weights.tolist()}")

    # Model
    model = UNet1D(
        in_channels=in_channels,
        num_classes=cfg.NUM_CLASSES,
        base_filters=cfg.BASE_FILTERS,
        depth=cfg.DEPTH,
        kernel_size=cfg.KERNEL_SIZE,
        dropout=cfg.DROPOUT,
    ).to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Input channels: {in_channels}, Loss: {cfg.LOSS_FUNCTION}, Dropout: {cfg.DROPOUT}")

    # Loss and optimizer
    if cfg.LOSS_FUNCTION == "focal":
        criterion = FocalLoss(alpha=class_weights, gamma=cfg.FOCAL_GAMMA)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.LEARNING_RATE, epochs=cfg.EPOCHS,
        steps_per_epoch=len(train_loader),
    )

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_f1 = 0.0
    best_epoch = 0
    best_metrics = {}
    patience_counter = 0
    history = []
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"Training: {cfg.EPOCHS} epochs, patience={cfg.PATIENCE}")
    print(f"{'='*60}\n")

    for epoch in range(1, cfg.EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)

        val_metrics = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_accuracy": val_metrics["accuracy"],
        })

        f1_str = "/".join(f"{f:.3f}" for f in val_metrics["f1_per_class"])
        print(
            f"Epoch {epoch:3d}/{cfg.EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val F1: {val_metrics['f1_macro']:.4f} ({f1_str}) | "
            f"Acc: {val_metrics['accuracy']:.4f} | "
            f"{elapsed:.1f}s"
        )

        # Early stopping on val F1 macro
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_epoch = epoch
            best_metrics = val_metrics
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": {k: v for k, v in vars(cfg).items() if k.isupper()},
                "train_ids": train_ids,
                "val_ids": val_ids,
            }, output_dir / "best_model.pt")
            print(f"  → Saved best model (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (patience={cfg.PATIENCE})")
                break

    total_time = time.time() - start_time

    # Save final model
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "val_metrics": val_metrics,
        "config": {k: v for k, v in vars(cfg).items() if k.isupper()},
        "history": history,
    }, output_dir / "final_model.pt")

    # ─── Write manifest.txt ───────────────────────────────────────────────
    manifest_path = output_dir / "manifest.txt"
    with open(manifest_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("AUTO-LABELER TRAINING MANIFEST\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data directory: {args.data_dir}\n")
        f.write(f"Output directory: {args.output_dir}\n")
        f.write(f"Device: {device}\n")
        if device.type == "cuda":
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
        f.write("\n")

        f.write("-" * 60 + "\n")
        f.write("PARAMETERS (from config.py)\n")
        f.write("-" * 60 + "\n")
        f.write(f"  SAMPLING_RATE_HZ      = {cfg.SAMPLING_RATE_HZ}\n")
        f.write(f"  NUM_CLASSES           = {cfg.NUM_CLASSES}\n")
        f.write(f"  NUM_CHANNELS          = {cfg.NUM_CHANNELS}\n")
        f.write(f"  CHUNK_SIZE            = {cfg.CHUNK_SIZE}\n")
        f.write(f"  BASE_FILTERS          = {cfg.BASE_FILTERS}\n")
        f.write(f"  DEPTH                 = {cfg.DEPTH}\n")
        f.write(f"  KERNEL_SIZE           = {cfg.KERNEL_SIZE}\n")
        f.write(f"  DROPOUT               = {cfg.DROPOUT}\n")
        f.write(f"  EPOCHS                = {cfg.EPOCHS}\n")
        f.write(f"  BATCH_SIZE            = {cfg.BATCH_SIZE}\n")
        f.write(f"  LEARNING_RATE         = {cfg.LEARNING_RATE}\n")
        f.write(f"  WEIGHT_DECAY          = {cfg.WEIGHT_DECAY}\n")
        f.write(f"  LOSS_FUNCTION         = {cfg.LOSS_FUNCTION}\n")
        f.write(f"  FOCAL_GAMMA           = {cfg.FOCAL_GAMMA}\n")
        f.write(f"  PATIENCE              = {cfg.PATIENCE}\n")
        f.write(f"  VAL_FRACTION          = {cfg.VAL_FRACTION}\n")
        f.write(f"  SEED                  = {cfg.SEED}\n")
        f.write(f"  NUM_WORKERS           = {cfg.NUM_WORKERS}\n")
        f.write(f"  GRAD_CLIP_NORM        = {cfg.GRAD_CLIP_NORM}\n")
        f.write(f"  TRAIN_STRIDE          = {stride}\n")
        f.write(f"  AUGMENT_NOISE_STD     = {cfg.AUGMENT_NOISE_STD}\n")
        f.write(f"  AUGMENT_SCALE_RANGE   = {cfg.AUGMENT_SCALE_RANGE}\n")
        f.write(f"  AUGMENT_OFFSET_RANGE  = {cfg.AUGMENT_OFFSET_RANGE}\n")
        f.write(f"  MIN_EVENT_DURATION_SEC= {cfg.MIN_EVENT_DURATION_SEC}\n")
        f.write("\n")

        f.write("-" * 60 + "\n")
        f.write(f"TRAINING FILES ({len(train_ids)} studies)\n")
        f.write("-" * 60 + "\n")
        for sid in sorted(train_ids):
            f.write(f"  {sid}\n")
        f.write("\n")

        f.write("-" * 60 + "\n")
        f.write(f"VALIDATION FILES ({len(val_ids)} studies)\n")
        f.write("-" * 60 + "\n")
        for sid in sorted(val_ids):
            f.write(f"  {sid}\n")
        f.write("\n")

        f.write("-" * 60 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("-" * 60 + "\n")
        f.write(f"  Model parameters      = {count_parameters(model):,}\n")
        f.write(f"  Class weights         = {class_weights.tolist()}\n")
        f.write(f"  Training chunks       = {len(train_ds):,}\n")
        f.write(f"  Validation chunks     = {len(val_ds):,}\n")
        f.write(f"  Total training time   = {total_time:.1f}s ({total_time/60:.1f} min)\n")
        f.write(f"  Epochs completed      = {epoch}\n")
        f.write(f"  Best epoch            = {best_epoch}\n")
        f.write(f"  Best val F1 macro     = {best_f1:.4f}\n")
        f.write(f"  Best val accuracy     = {best_metrics.get('accuracy', 0):.4f}\n")
        f.write(f"  Best val F1 per class:\n")
        for i, name in enumerate(cfg.CLASS_NAMES):
            f1_val = best_metrics.get("f1_per_class", [0, 0, 0])[i]
            f.write(f"    {name:10s} = {f1_val:.4f}\n")
        f.write(f"  Final train loss      = {history[-1]['train_loss']:.4f}\n")
        f.write(f"  Final val loss        = {history[-1]['val_loss']:.4f}\n")
        f.write("\n" + "=" * 60 + "\n")

    print(f"\n{'='*60}")
    print(f"Training complete. Best val F1: {best_f1:.4f} (epoch {best_epoch})")
    print(f"Checkpoints saved to: {output_dir}")
    print(f"Manifest saved to: {manifest_path}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
