"""FarmWatch Dataset Preparation & Split Generator.

Audits images, groups duplicates (using SHA-256 so no duplicate crosses splits),
creates stratified deterministic Train (70%) / Val (15%) / Test (15%) splits,
writes data/processed/splits.csv, and optionally materializes folder splits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image

CANONICAL_CLASSES = ["bare_soil", "healthy", "high_stress", "low_stress", "rust"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def calculate_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def prepare_dataset(
    source_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    materialize_folders: bool = True,
) -> Path:
    """Prepares deterministic split manifest and materialized split directories."""
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "splits.csv"

    print(f"[INFO] Scanning images in {source_dir}...")
    # Map class -> list of unique items (grouped by SHA-256 to eliminate cross-split duplicate leakage)
    class_groups: Dict[str, Dict[str, List[Path]]] = defaultdict(lambda: defaultdict(list))
    total_images = 0

    for root, _, files in os.walk(source_dir):
        for f in sorted(files):
            full_path = Path(root) / f
            if full_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            parent_name = full_path.parent.name.lower()
            if parent_name not in CANONICAL_CLASSES:
                continue

            # Verify image can be opened
            try:
                with Image.open(full_path) as img:
                    img.verify()
            except Exception as e:
                print(f"[WARN] Skipping corrupt image {full_path}: {e}")
                continue

            sha = calculate_sha256(full_path)
            class_groups[parent_name][sha].append(full_path)
            total_images += 1

    print(f"[INFO] Found {total_images} valid images across {len(class_groups)} canonical classes.")

    records: List[Dict[str, str]] = []
    split_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    global_hash_to_split: Dict[str, str] = {}

    for cls in CANONICAL_CLASSES:
        groups = class_groups[cls]
        unique_hashes = sorted(groups.keys())
        unassigned_hashes = [h for h in unique_hashes if h not in global_hash_to_split]

        # Deterministic shuffle of unique groups
        rng = random.Random(seed + hash(cls))
        rng.shuffle(unassigned_hashes)

        n_groups = len(unassigned_hashes)
        n_train = max(1, int(round(n_groups * train_ratio)))
        n_val = max(1, int(round(n_groups * val_ratio)))

        for h in unassigned_hashes[:n_train]:
            global_hash_to_split[h] = "train"
        for h in unassigned_hashes[n_train : n_train + n_val]:
            global_hash_to_split[h] = "val"
        for h in unassigned_hashes[n_train + n_val :]:
            global_hash_to_split[h] = "test"

        for sha, paths in groups.items():
            split = global_hash_to_split[sha]
            for p in paths:
                rel_path = p.relative_to(source_dir).as_posix()
                records.append({
                    "relative_path": rel_path,
                    "source_label": cls,
                    "canonical_label": cls,
                    "split": split,
                    "group_id": sha[:12],
                    "sha256": sha,
                })
                split_counts[split][cls] += 1

    # Sort records for reproducible output
    records.sort(key=lambda x: (x["split"], x["canonical_label"], x["relative_path"]))

    # Write manifest CSV
    fieldnames = ["relative_path", "source_label", "canonical_label", "split", "group_id", "sha256"]
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"[SUCCESS] Split manifest written to {manifest_path} ({len(records)} entries)")

    # Print summary statistics
    print("\n--- SPLIT DISTRIBUTION SUMMARY ---")
    splits = ["train", "val", "test"]
    header = f"{'Class':<15} | " + " | ".join(f"{s.capitalize():<8}" for s in splits) + " | Total"
    print(header)
    print("-" * len(header))
    for cls in CANONICAL_CLASSES:
        row = f"{cls:<15} | "
        total_cls = 0
        for s in splits:
            cnt = split_counts[s][cls]
            total_cls += cnt
            row += f"{cnt:<8} | "
        row += f"{total_cls:<5}"
        print(row)

    print("-" * len(header))
    total_row = f"{'TOTAL':<15} | "
    grand_total = 0
    for s in splits:
        s_total = sum(split_counts[s].values())
        grand_total += s_total
        total_row += f"{s_total:<8} | "
    total_row += f"{grand_total:<5}"
    print(total_row)

    # Materialize folder layout if requested
    if materialize_folders:
        print("\n[INFO] Materializing split folders in data/processed/...")
        for rec in records:
            src_file = source_dir / rec["relative_path"]
            dest_file = output_dir / rec["split"] / rec["canonical_label"] / Path(rec["relative_path"]).name
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            if not dest_file.exists():
                shutil.copy2(src_file, dest_file)
        print("[SUCCESS] Materialized train/, val/, and test/ directories.")

    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="FarmWatch Dataset Split Generator")
    parser.add_argument(
        "--source-dir",
        type=str,
        default="data/raw/extracted/mini_datasets/mini_dataset_3ch",
        help="Source directory containing class folders",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for processed splits and splits.csv",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train ratio (default: 0.70)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio (default: 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splits (default: 42)")
    parser.add_argument("--no-folders", action="store_true", help="Skip materializing folder directories")

    args = parser.parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)

    if not source_dir.is_dir():
        print(f"[ERROR] Source directory does not exist: {source_dir}", file=sys.stderr)
        return 1

    try:
        prepare_dataset(
            source_dir=source_dir,
            output_dir=output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            materialize_folders=not args.no_folders,
        )
    except Exception as e:
        print(f"[ERROR] Dataset preparation failed: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
