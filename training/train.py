"""FarmWatch ResNet-18 Transfer Learning Training Pipeline.

Executes two-stage transfer learning:
1. Frozen backbone, head-only training.
2. Fine-tuning of layer4 + classification head.
Selects best checkpoint by validation macro F1 and exports history + curves.
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.model_utils import (
    DEFAULT_IMAGE_SIZE,
    FarmWatchDataset,
    build_resnet18,
    get_transforms,
    load_split_records,
    save_checkpoint,
    set_backbone_trainable,
)


def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    is_train: bool = True,
) -> Tuple[float, float, float]:
    """Run single training or validation epoch, returning loss, accuracy, and macro F1."""
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            if is_train and optimizer:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, targets)

            if is_train and optimizer:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    epoch_loss = total_loss / len(loader.dataset)
    acc = float(np.mean(np.array(all_preds) == np.array(all_targets)))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))

    return epoch_loss, acc, macro_f1


def plot_training_curves(history: List[Dict[str, Any]], output_path: Path) -> None:
    """Plot and save training and validation loss and macro F1 curves."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_f1 = [h["val_macro_f1"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss subplot
    ax1.plot(epochs, train_loss, label="Train Loss", marker="o", color="#2d6a4f")
    ax1.plot(epochs, val_loss, label="Validation Loss", marker="s", color="#d90429")
    ax1.set_title("Cross-Entropy Loss Across Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Macro F1 subplot
    ax2.plot(epochs, val_f1, label="Val Macro F1", marker="^", color="#1b4332")
    ax2.set_title("Validation Macro F1 (Model Selection Metric)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro F1")
    ax2.set_ylim([0.0, 1.05])
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[INFO] Training curves saved to: {output_path}")


def train_farmwatch(
    manifest_path: Path,
    data_dir: Path,
    output_model_path: Path,
    output_reports_dir: Path,
    epochs_head: int = 5,
    epochs_finetune: int = 8,
    batch_size: int = 16,
    lr_head: float = 1e-3,
    lr_finetune: float = 1e-4,
    seed: int = 42,
    device_name: str = "auto",
) -> Path:
    """Executes full staged training and checkpoint export."""
    set_seed(seed)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print(f"[INFO] Using compute device: {device}")

    # Load dataset splits
    train_recs, class_names, class_to_idx = load_split_records(manifest_path, "train")
    val_recs, _, _ = load_split_records(manifest_path, "val")

    print(f"[INFO] Classes ({len(class_names)}): {class_names}")
    print(f"[INFO] Training samples: {len(train_recs)} | Validation samples: {len(val_recs)}")

    train_tf, eval_tf = get_transforms(DEFAULT_IMAGE_SIZE)

    train_ds = FarmWatchDataset(data_dir, train_recs, class_to_idx, transform=train_tf)
    val_ds = FarmWatchDataset(data_dir, val_recs, class_to_idx, transform=eval_tf)

    # Windows safe workers: 0 for compatibility
    num_workers = 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Initialize model
    print("[INFO] Loading ImageNet pre-trained ResNet-18...")
    model = build_resnet18(num_classes=len(class_names), pretrained=True, dropout=0.2).to(device)
    criterion = nn.CrossEntropyLoss()

    history: List[Dict[str, Any]] = []
    best_val_f1 = -1.0
    best_model_weights = None
    best_epoch = 0

    total_epoch = 0

    # Stage 1: Head-Only
    if epochs_head > 0:
        print(f"\n=== STAGE 1: Head-Only Training ({epochs_head} epochs, lr={lr_head}) ===")
        set_backbone_trainable(model, "head_only")
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_head, weight_decay=1e-4)

        for ep in range(1, epochs_head + 1):
            total_epoch += 1
            t_loss, t_acc, t_f1 = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
            v_loss, v_acc, v_f1 = run_epoch(model, val_loader, criterion, None, device, is_train=False)

            print(f"Epoch {total_epoch:02d} (Stage 1.{ep}) | Train Loss: {t_loss:.4f} Acc: {t_acc:.4f} | Val Loss: {v_loss:.4f} Acc: {v_acc:.4f} Macro F1: {v_f1:.4f}")

            history.append({
                "epoch": total_epoch,
                "stage": "head_only",
                "train_loss": round(t_loss, 4),
                "train_acc": round(t_acc, 4),
                "train_macro_f1": round(t_f1, 4),
                "val_loss": round(v_loss, 4),
                "val_acc": round(v_acc, 4),
                "val_macro_f1": round(v_f1, 4),
            })

            if v_f1 > best_val_f1:
                best_val_f1 = v_f1
                best_epoch = total_epoch
                best_model_weights = copy.deepcopy(model.state_dict())
                print(f"  --> [NEW BEST] Validation Macro F1: {best_val_f1:.4f} (Saved weights)")

    # Stage 2: Fine-Tuning layer4 + head
    if epochs_finetune > 0:
        print(f"\n=== STAGE 2: Fine-Tuning Layer4 + Head ({epochs_finetune} epochs, lr={lr_finetune}) ===")
        set_backbone_trainable(model, "layer4_and_head")
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_finetune, weight_decay=1e-4)

        for ep in range(1, epochs_finetune + 1):
            total_epoch += 1
            t_loss, t_acc, t_f1 = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
            v_loss, v_acc, v_f1 = run_epoch(model, val_loader, criterion, None, device, is_train=False)

            print(f"Epoch {total_epoch:02d} (Stage 2.{ep}) | Train Loss: {t_loss:.4f} Acc: {t_acc:.4f} | Val Loss: {v_loss:.4f} Acc: {v_acc:.4f} Macro F1: {v_f1:.4f}")

            history.append({
                "epoch": total_epoch,
                "stage": "layer4_finetune",
                "train_loss": round(t_loss, 4),
                "train_acc": round(t_acc, 4),
                "train_macro_f1": round(t_f1, 4),
                "val_loss": round(v_loss, 4),
                "val_acc": round(v_acc, 4),
                "val_macro_f1": round(v_f1, 4),
            })

            if v_f1 > best_val_f1:
                best_val_f1 = v_f1
                best_epoch = total_epoch
                best_model_weights = copy.deepcopy(model.state_dict())
                print(f"  --> [NEW BEST] Validation Macro F1: {best_val_f1:.4f} (Saved weights)")

    # Restore best weights
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        print(f"\n[INFO] Loaded best checkpoint weights from Epoch {best_epoch} with Val Macro F1: {best_val_f1:.4f}")

    # Save Checkpoint
    training_config = {
        "epochs_head": epochs_head,
        "epochs_finetune": epochs_finetune,
        "batch_size": batch_size,
        "lr_head": lr_head,
        "lr_finetune": lr_finetune,
        "seed": seed,
        "device": str(device),
        "best_epoch": best_epoch,
    }

    save_checkpoint(
        path=output_model_path,
        model=model,
        class_names=class_names,
        image_size=DEFAULT_IMAGE_SIZE,
        best_f1=best_val_f1,
        training_config=training_config,
    )
    print(f"[SUCCESS] Checkpoint saved to: {output_model_path}")

    # Export history CSV
    output_reports_dir.mkdir(parents=True, exist_ok=True)
    history_csv = output_reports_dir / "training_history.csv"
    with open(history_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(f"[SUCCESS] Training history written to: {history_csv}")

    # Plot curves
    curves_png = output_reports_dir / "training_curves.png"
    plot_training_curves(history, curves_png)

    return output_model_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Train FarmWatch ResNet-18 Baseline")
    parser.add_argument("--manifest", type=str, default="data/processed/splits.csv", help="Path to splits manifest")
    parser.add_argument("--data-dir", type=str, default="data/raw/extracted/mini_datasets/mini_dataset_3ch", help="Image directory")
    parser.add_argument("--output-model", type=str, default="artifacts/models/farmwatch_resnet18.pth", help="Checkpoint target path")
    parser.add_argument("--reports-dir", type=str, default="artifacts/reports", help="Reports target directory")
    parser.add_argument("--epochs-head", type=int, default=5, help="Stage 1 head-only epochs (default: 5)")
    parser.add_argument("--epochs-finetune", type=int, default=6, help="Stage 2 fine-tuning epochs (default: 6)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr-head", type=float, default=1e-3, help="Head learning rate")
    parser.add_argument("--lr-finetune", type=float, default=1e-4, help="Fine-tuning learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, auto)")

    args = parser.parse_args()

    try:
        train_farmwatch(
            manifest_path=Path(args.manifest),
            data_dir=Path(args.data_dir),
            output_model_path=Path(args.output_model),
            output_reports_dir=Path(args.reports_dir),
            epochs_head=args.epochs_head,
            epochs_finetune=args.epochs_finetune,
            batch_size=args.batch_size,
            lr_head=args.lr_head,
            lr_finetune=args.lr_finetune,
            seed=args.seed,
            device_name=args.device,
        )
    except Exception as e:
        print(f"[ERROR] Training failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
