"""FarmWatch Dataset Verification & Duplicate Audit Tool.

Scans candidate images, verifies integrity with Pillow,
detects exact SHA-256 duplicates and perceptual (dHash) near-duplicates,
and exports artifacts/reports/dataset_verification.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CANONICAL_CLASSES = {"healthy", "low_stress", "high_stress", "rust", "bare_soil"}


def calculate_sha256(path: Path) -> str:
    """Calculate exact SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def calculate_dhash(img: Image.Image, hash_size: int = 8) -> str:
    """Calculate difference hash (dHash) for perceptual similarity detection."""
    # Resize to (hash_size + 1, hash_size) grayscale
    resized = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(resized.getdata())
    # Compare adjacent pixels
    diff = []
    for row in range(hash_size):
        row_offset = row * (hash_size + 1)
        for col in range(hash_size):
            pixel_left = pixels[row_offset + col]
            pixel_right = pixels[row_offset + col + 1]
            diff.append(pixel_left > pixel_right)
    # Convert binary list to hex string
    decimal_val = 0
    hex_str = []
    for index, val in enumerate(diff):
        if val:
            decimal_val += 1 << (index % 4)
        if index % 4 == 3:
            hex_str.append(hex(decimal_val)[2:])
            decimal_val = 0
    return "".join(hex_str)


def verify_dataset(data_dir: Path) -> Dict[str, Any]:
    """Verify all images in dataset directory."""
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Directory not found: {data_dir}")

    total_scanned = 0
    valid_count = 0
    corrupt_files: List[Dict[str, str]] = []
    sha256_map: Dict[str, List[str]] = defaultdict(list)
    dhash_map: Dict[str, List[str]] = defaultdict(list)
    class_counts: Dict[str, int] = defaultdict(int)
    dimensions_count: Dict[str, int] = defaultdict(int)
    modes_count: Dict[str, int] = defaultdict(int)

    image_records: List[Dict[str, Any]] = []

    for root, _, files in os.walk(data_dir):
        for f in sorted(files):
            full_path = Path(root) / f
            ext = full_path.suffix.lower()
            if ext not in IMAGE_EXTENSIONS:
                continue

            total_scanned += 1
            rel_path = full_path.relative_to(data_dir).as_posix()
            parent_name = full_path.parent.name.lower()

            canonical = None
            for c in CANONICAL_CLASSES:
                if c == parent_name:
                    canonical = c
                    break

            # 1. Pillow verify and decode
            try:
                with open(full_path, "rb") as fp:
                    with Image.open(fp) as img:
                        img.verify()
                # Reopen to read dimensions, mode, and calculate dHash
                with Image.open(full_path) as img:
                    mode = img.mode
                    size = img.size
                    dh = calculate_dhash(img)

                sha = calculate_sha256(full_path)
                sha256_map[sha].append(rel_path)
                dhash_map[dh].append(rel_path)

                valid_count += 1
                if canonical:
                    class_counts[canonical] += 1
                else:
                    class_counts[f"unmapped_{parent_name}"] += 1

                dimensions_count[f"{size[0]}x{size[1]}"] += 1
                modes_count[mode] += 1

                image_records.append({
                    "relative_path": rel_path,
                    "class": canonical or parent_name,
                    "sha256": sha,
                    "dhash": dh,
                    "size": size,
                    "mode": mode,
                })

            except Exception as e:
                corrupt_files.append({"file": rel_path, "error": str(e)})

    # Detect duplicates
    exact_duplicates = {sha: paths for sha, paths in sha256_map.items() if len(paths) > 1}
    perceptual_duplicates = {dh: paths for dh, paths in dhash_map.items() if len(paths) > 1}

    return {
        "dataset_directory": str(data_dir),
        "total_images_scanned": total_scanned,
        "valid_images": valid_count,
        "corrupted_images_count": len(corrupt_files),
        "corrupted_files": corrupt_files,
        "class_distribution": dict(class_counts),
        "dimension_distribution": dict(dimensions_count),
        "mode_distribution": dict(modes_count),
        "exact_duplicate_groups": len(exact_duplicates),
        "exact_duplicate_samples": exact_duplicates,
        "perceptual_duplicate_groups": len(perceptual_duplicates),
        "perceptual_duplicate_samples": {dh: paths for dh, paths in list(perceptual_duplicates.items())[:5]},
        "records_count": len(image_records),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="FarmWatch Dataset Verification & Duplicate Audit")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/extracted/mini_datasets/mini_dataset_3ch",
        help="Path to dataset class directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/reports/dataset_verification.json",
        help="Path to output JSON report",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print(f"[INFO] Verifying dataset at: {data_dir}")

    try:
        report = verify_dataset(data_dir)
    except Exception as e:
        print(f"[ERROR] Verification failed: {e}", file=sys.stderr)
        return 1

    print("\n--- DATASET VERIFICATION RESULTS ---")
    print(f"Total Scanned: {report['total_images_scanned']}")
    print(f"Valid Images:  {report['valid_images']}")
    print(f"Corrupted:     {report['corrupted_images_count']}")
    print(f"Exact Duplicate Groups (SHA-256): {report['exact_duplicate_groups']}")
    print(f"Perceptual Duplicate Groups (dHash): {report['perceptual_duplicate_groups']}")

    print("\nClass Distribution:")
    for cls, cnt in report["class_distribution"].items():
        print(f"  {cls}: {cnt}")

    print("\nImage Dimensions:", report["dimension_distribution"])
    print("Colour Modes:", report["mode_distribution"])

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[SUCCESS] Verification report written to: {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
