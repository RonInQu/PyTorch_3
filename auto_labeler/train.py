"""
Training script for the 1D U-Net auto-labeler.

Usage:
    python -m auto_labeler.train --data_dir training_data --epochs 80
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import (
    CHUNK_SIZE,
    NUM_CLASSES,
    TRAINING_STUDIES,
    SegmentationDataset,
    compute_class_weights,
    create_datasets,
)
from .model import UNet1D, count_parameters


def compute_metrics(preds: torch.Tensor, labels: torch.Tensor, num_classes: int = NUM_CLASSES):
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
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    parser.add_argument("--data_dir", type=str, default="training_data")
    parser.add_argument("--output_dir", type=str, default="auto_labeler/checkpoints")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--base_filters", type=int, default=32)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--kernel_size", type=int, default=7)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create datasets
    print("Loading data...")
    train_ds, val_ds, train_ids, val_ids = create_datasets(
        data_dir=args.data_dir,
        val_fraction=args.val_fraction,
        chunk_size=args.chunk_size,
        stride=args.stride,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    # Class weights
    print("Computing class weights...")
    class_weights = compute_class_weights(Path(args.data_dir), train_ids).to(device)
    print(f"Class weights: {class_weights.tolist()}")

    # Model
    model = UNet1D(
        in_channels=1,
        num_classes=NUM_CLASSES,
        base_filters=args.base_filters,
        depth=args.depth,
        kernel_size=args.kernel_size,
    ).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=len(train_loader),
    )

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_f1 = 0.0
    patience_counter = 0
    history = []

    print(f"\n{'='*60}")
    print(f"Training: {args.epochs} epochs, patience={args.patience}")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
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
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val F1: {val_metrics['f1_macro']:.4f} ({f1_str}) | "
            f"Acc: {val_metrics['accuracy']:.4f} | "
            f"{elapsed:.1f}s"
        )

        # Early stopping on val F1 macro
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "args": vars(args),
                "train_ids": train_ids,
                "val_ids": val_ids,
            }, output_dir / "best_model.pt")
            print(f"  → Saved best model (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
                break

    # Save final model
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "val_metrics": val_metrics,
        "args": vars(args),
        "history": history,
    }, output_dir / "final_model.pt")

    print(f"\n{'='*60}")
    print(f"Training complete. Best val F1: {best_f1:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
