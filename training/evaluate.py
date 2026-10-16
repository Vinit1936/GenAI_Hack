"""FarmWatch Model Evaluation Tool.

Loads the saved best checkpoint, evaluates strictly on the held-out test split,
computes overall accuracy, macro precision/recall/F1, per-class metrics,
confusion matrix, and exports results to artifacts/reports/.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.model_utils import (
    DEFAULT_IMAGE_SIZE,
    FarmWatchDataset,
    get_transforms,
    load_checkpoint,
    load_split_records,
)


def evaluate_checkpoint(
    checkpoint_path: Path,
    manifest_path: Path,
    data_dir: Path,
    reports_dir: Path,
    batch_size: int = 16,
    device_name: str = "auto",
) -> Dict[str, Any]:
    """Evaluates checkpoint on the held-out test set without tuning."""
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print(f"[INFO] Loading checkpoint from: {checkpoint_path}")
    model, class_names, ckpt_meta = load_checkpoint(checkpoint_path, device)
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    print(f"[INFO] Loaded architecture: {ckpt_meta.get('architecture')}")
    print(f"[INFO] Class order: {class_names}")

    test_recs, _, _ = load_split_records(manifest_path, "test")
    print(f"[INFO] Evaluating strictly on held-out test set: {len(test_recs)} samples")

    _, eval_tf = get_transforms(DEFAULT_IMAGE_SIZE)
    test_ds = FarmWatchDataset(data_dir, test_recs, class_to_idx, transform=eval_tf)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    model.eval()
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    test_loss = total_loss / len(test_ds)
    all_targets_np = np.array(all_targets)
    all_preds_np = np.array(all_preds)

    acc = float(np.mean(all_preds_np == all_targets_np))
    macro_precision = float(precision_score(all_targets_np, all_preds_np, average="macro", zero_division=0))
    macro_recall = float(recall_score(all_targets_np, all_preds_np, average="macro", zero_division=0))
    macro_f1 = float(f1_score(all_targets_np, all_preds_np, average="macro", zero_division=0))

    # Per-class metrics
    clf_dict = classification_report(
        all_targets_np,
        all_preds_np,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # Confusion matrix
    cm = confusion_matrix(all_targets_np, all_preds_np, labels=list(range(len(class_names))))

    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title=f"Test Set Confusion Matrix (Macro F1: {macro_f1:.4f})",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    cm_path = reports_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[INFO] Confusion matrix saved to: {cm_path}")

    # 2. Save CSV classification report
    csv_path = reports_dir / "classification_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1-score", "support"])
        for k, v in clf_dict.items():
            if isinstance(v, dict):
                writer.writerow([k, f"{v['precision']:.4f}", f"{v['recall']:.4f}", f"{v['f1-score']:.4f}", v["support"]])
            else:
                writer.writerow([k, f"{v:.4f}", "", "", ""])
    print(f"[INFO] Classification report saved to: {csv_path}")

    # 3. Save JSON evaluation summary
    results = {
        "model_architecture": ckpt_meta.get("architecture"),
        "model_version": ckpt_meta.get("model_version"),
        "test_sample_count": len(test_recs),
        "test_loss": round(test_loss, 4),
        "overall_accuracy": round(acc, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_metrics": {cls: clf_dict[cls] for cls in class_names if cls in clf_dict},
        "confusion_matrix": cm.tolist(),
        "class_order": class_names,
        "methodology_notes": (
            "Evaluated strictly on held-out test split (15% of dataset). "
            "Exact SHA-256 duplicate clusters were isolated into individual partitions "
            "prior to splitting to prevent cross-partition leakage."
        ),
    }

    json_path = reports_dir / "evaluation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Evaluation results JSON saved to: {json_path}")

    print("\n--- FINAL TEST EVALUATION METRICS ---")
    print(f"Test Samples:     {len(test_recs)}")
    print(f"Overall Accuracy: {acc:.4%}")
    print(f"Macro Precision:  {macro_precision:.4f}")
    print(f"Macro Recall:     {macro_recall:.4f}")
    print(f"Macro F1 Score:   {macro_f1:.4f}")
    print(f"Test Loss:        {test_loss:.4f}")

    print("\nPer-Class Breakdown:")
    for cls in class_names:
        m = clf_dict[cls]
        print(f"  {cls:<12} | Precision: {m['precision']:.4f} | Recall: {m['recall']:.4f} | F1: {m['f1-score']:.4f} | Support: {int(m['support'])}")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate FarmWatch Checkpoint")
    parser.add_argument("--checkpoint", type=str, default="artifacts/models/farmwatch_resnet18.pth", help="Checkpoint path")
    parser.add_argument("--manifest", type=str, default="data/processed/splits.csv", help="Splits manifest path")
    parser.add_argument("--data-dir", type=str, default="data/raw/extracted/mini_datasets/mini_dataset_3ch", help="Image directory")
    parser.add_argument("--reports-dir", type=str, default="artifacts/reports", help="Reports output directory")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, auto)")

    args = parser.parse_args()

    try:
        evaluate_checkpoint(
            checkpoint_path=Path(args.checkpoint),
            manifest_path=Path(args.manifest),
            data_dir=Path(args.data_dir),
            reports_dir=Path(args.reports_dir),
            batch_size=args.batch_size,
            device_name=args.device,
        )
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
