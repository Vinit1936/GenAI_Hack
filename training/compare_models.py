"""FarmWatch Architecture Comparison & Benchmark Utility.

Compares ResNet-18 baseline against alternative compact vision backbones
(e.g., MobileNetV3-Small) on identical held-out test splits.
Evaluates:
  - Macro F1 score, Accuracy, Precision, Recall
  - Model parameter count and checkpoint storage footprint
  - Average CPU inference latency per patch (ms)
Exports structured comparison report to artifacts/reports/model_comparison.json and .csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from training.model_utils import (
    FarmWatchDataset,
    get_transforms,
    load_checkpoint,
    load_split_records,
)


def evaluate_model_performance(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluates accuracy, macro precision, recall, F1, and avg inference latency."""
    model.eval()
    all_preds = []
    all_targets = []
    latencies = []

    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device)

            start = time.perf_counter()
            outputs = model(images)
            latencies.append((time.perf_counter() - start) / len(images))

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

    acc = float(accuracy_score(all_targets, all_preds))
    prec = float(precision_score(all_targets, all_preds, average="macro", zero_division=0))
    rec = float(recall_score(all_targets, all_preds, average="macro", zero_division=0))
    f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    avg_latency_ms = float(np.mean(latencies) * 1000.0)

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(prec, 4),
        "macro_recall": round(rec, 4),
        "macro_f1": round(f1, 4),
        "avg_latency_ms": round(avg_latency_ms, 2),
    }


def count_parameters(model: nn.Module) -> int:
    """Counts total trainable and non-trainable model parameters."""
    return sum(p.numel() for p in model.parameters())


def run_model_comparison(
    resnet_checkpoint: Path,
    splits_csv: Path,
    output_dir: Path,
    device_name: str = "auto",
) -> Dict[str, Any]:
    """Runs end-to-end benchmark comparison between ResNet-18 and MobileNetV3."""
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print(f"[INFO] Running model benchmark comparison on device: {device}")

    # 1. Load checkpoint to get classes contract
    resnet_model, class_names, metadata = load_checkpoint(resnet_checkpoint, device)
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    # 2. Load held-out test split
    data_dir = Path("data/raw/extracted/mini_datasets/mini_dataset_3ch")
    test_records, _, _ = load_split_records(splits_csv, "test")
    from training.model_utils import DEFAULT_IMAGE_SIZE
    _, eval_tf = get_transforms(DEFAULT_IMAGE_SIZE)
    test_dataset = FarmWatchDataset(data_dir, test_records, class_to_idx, transform=eval_tf)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    print(f"[INFO] Loaded {len(test_dataset)} test samples across {len(class_names)} classes.")

    # 3. Evaluate ResNet-18 Checkpoint
    resnet_params = count_parameters(resnet_model)
    resnet_size_mb = round(resnet_checkpoint.stat().st_size / (1024 * 1024), 2)
    resnet_metrics = evaluate_model_performance(resnet_model, test_loader, device)

    resnet_entry = {
        "architecture": "ResNet-18",
        "parameters": resnet_params,
        "checkpoint_size_mb": resnet_size_mb,
        **resnet_metrics,
    }
    print(f"[INFO] ResNet-18 -> Accuracy: {resnet_entry['accuracy']:.2%}, Macro F1: {resnet_entry['macro_f1']:.4f}, Latency: {resnet_entry['avg_latency_ms']:.2f}ms")

    # 3. Evaluate MobileNetV3-Small Reference Baseline
    mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    in_features = mobilenet.classifier[3].in_features
    mobilenet.classifier[3] = nn.Linear(in_features, len(class_names))
    mobilenet = mobilenet.to(device)

    mobilenet_params = count_parameters(mobilenet)
    # Estimate size in MB (float32 = 4 bytes per param)
    mobilenet_est_size_mb = round(mobilenet_params * 4 / (1024 * 1024), 2)
    mobilenet_metrics = evaluate_model_performance(mobilenet, test_loader, device)

    mobilenet_entry = {
        "architecture": "MobileNetV3-Small (Pretrained Ref)",
        "parameters": mobilenet_params,
        "checkpoint_size_mb": mobilenet_est_size_mb,
        **mobilenet_metrics,
    }
    print(f"[INFO] MobileNetV3-Small -> Params: {mobilenet_params:,}, Est Size: {mobilenet_est_size_mb}MB, Latency: {mobilenet_entry['avg_latency_ms']:.2f}ms")

    results = {
        "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": str(device),
        "test_samples": len(test_dataset),
        "models": [resnet_entry, mobilenet_entry],
    }

    # 4. Export JSON & CSV Reports
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "model_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    csv_path = output_dir / "model_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["architecture", "parameters", "checkpoint_size_mb", "accuracy", "macro_precision", "macro_recall", "macro_f1", "avg_latency_ms"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in results["models"]:
            writer.writerow(m)

    print(f"[SUCCESS] Benchmark reports written to '{json_path}' and '{csv_path}'.")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="FarmWatch Model Architecture Comparison Benchmark")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/models/farmwatch_resnet18.pth"))
    parser.add_argument("--splits", type=Path, default=Path("data/processed/splits.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reports"))
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()
    run_model_comparison(
        resnet_checkpoint=args.checkpoint,
        splits_csv=args.splits,
        output_dir=args.output_dir,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
