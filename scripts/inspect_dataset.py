"""FarmWatch Dataset Inspection Tool.

Inspects raw dataset directories, ZIP archives, or metadata files,
counts valid images, detects classes, checks channel modes and dimensions with Pillow,
and writes artifacts/reports/dataset_inspection.json.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CANONICAL_CLASSES = {"healthy", "low_stress", "high_stress", "rust", "bare_soil"}


def is_safe_zip_path(target_path: str) -> bool:
    """Check that a ZIP entry path is safe (no absolute paths or path traversal)."""
    p = Path(target_path)
    return not p.is_absolute() and ".." not in p.parts


def calculate_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """Calculate hash of a local file."""
    h = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def inspect_image_bytes(data: bytes, filename: str) -> Tuple[bool, Optional[str], Optional[Tuple[int, int]], Optional[str]]:
    """Decode image using Pillow and extract dimensions and mode."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
        with Image.open(io.BytesIO(data)) as img:
            mode = img.mode
            size = img.size
            return True, mode, size, None
    except Exception as e:
        return False, None, None, f"Decode failure: {str(e)}"


def inspect_zip(zip_path: Path) -> Dict[str, Any]:
    """Inspect contents of a ZIP archive safely without extracting."""
    if not zip_path.is_file():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    md5_hash = calculate_file_hash(zip_path, "md5")
    sha256_hash = calculate_file_hash(zip_path, "sha256")
    file_size = zip_path.stat().st_size

    extension_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    subfolder_counts: Counter[str] = Counter()
    corrupted_images: List[Dict[str, str]] = []
    modes: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()
    metadata_files: List[str] = []
    sample_files: List[str] = []

    with zipfile.ZipFile(zip_path, "r") as z:
        entries = z.infolist()
        total_entries = len(entries)

        for entry in entries:
            name = entry.filename
            if not is_safe_zip_path(name):
                continue
            if entry.is_dir():
                continue

            ext = Path(name).suffix.lower()
            extension_counts[ext] += 1

            if ext in {".csv", ".json", ".txt", ".md"}:
                metadata_files.append(name)

            if ext in IMAGE_EXTENSIONS:
                parts = Path(name).parts
                if len(parts) > 1:
                    parent_dir = parts[-2]
                    subfolder_counts[parent_dir] += 1
                    # Check if parent matches canonical or close match
                    for c in CANONICAL_CLASSES:
                        if c in parent_dir.lower():
                            class_counts[c] += 1
                            break

                if len(sample_files) < 10:
                    sample_files.append(name)

                # Decode a sample of up to 50 images to test integrity & get dimensions/mode
                if sum(modes.values()) < 50:
                    try:
                        data = z.read(entry)
                        valid, mode, size, err = inspect_image_bytes(data, name)
                        if valid and mode and size:
                            modes[mode] += 1
                            dimensions[f"{size[0]}x{size[1]}"] += 1
                        else:
                            corrupted_images.append({"file": name, "error": err or "Unknown decode error"})
                    except Exception as e:
                        corrupted_images.append({"file": name, "error": str(e)})

    return {
        "source_path": str(zip_path),
        "source_type": "zip_archive",
        "file_size_bytes": file_size,
        "md5": md5_hash,
        "sha256": sha256_hash,
        "total_entries": total_entries,
        "extensions": dict(extension_counts),
        "detected_subfolders": dict(subfolder_counts),
        "detected_classes": dict(class_counts),
        "sample_modes": dict(modes),
        "sample_dimensions": dict(dimensions),
        "metadata_files": metadata_files,
        "sample_paths": sample_files,
        "corrupted_images_detected": len(corrupted_images),
        "corrupted_samples": corrupted_images[:5],
    }


def inspect_directory(dir_path: Path) -> Dict[str, Any]:
    """Inspect contents of an unzipped directory."""
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Directory not found: {dir_path}")

    extension_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    subfolder_counts: Counter[str] = Counter()
    corrupted_images: List[Dict[str, str]] = []
    modes: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()
    metadata_files: List[str] = []
    sample_files: List[str] = []
    total_files = 0

    for root, _, files in os.walk(dir_path):
        for f in files:
            total_files += 1
            full_path = Path(root) / f
            rel_path = full_path.relative_to(dir_path)
            ext = full_path.suffix.lower()
            extension_counts[ext] += 1

            if ext in {".csv", ".json", ".txt", ".md"}:
                metadata_files.append(str(rel_path))

            if ext in IMAGE_EXTENSIONS:
                parts = rel_path.parts
                if len(parts) > 1:
                    parent_dir = parts[-2]
                    subfolder_counts[parent_dir] += 1
                    for c in CANONICAL_CLASSES:
                        if c in parent_dir.lower():
                            class_counts[c] += 1
                            break

                if len(sample_files) < 10:
                    sample_files.append(str(rel_path))

                if sum(modes.values()) < 50:
                    try:
                        with open(full_path, "rb") as img_file:
                            data = img_file.read()
                        valid, mode, size, err = inspect_image_bytes(data, str(rel_path))
                        if valid and mode and size:
                            modes[mode] += 1
                            dimensions[f"{size[0]}x{size[1]}"] += 1
                        else:
                            corrupted_images.append({"file": str(rel_path), "error": err or "Decode error"})
                    except Exception as e:
                        corrupted_images.append({"file": str(rel_path), "error": str(e)})

    return {
        "source_path": str(dir_path),
        "source_type": "directory",
        "total_files": total_files,
        "extensions": dict(extension_counts),
        "detected_subfolders": dict(subfolder_counts),
        "detected_classes": dict(class_counts),
        "sample_modes": dict(modes),
        "sample_dimensions": dict(dimensions),
        "metadata_files": metadata_files,
        "sample_paths": sample_files,
        "corrupted_images_detected": len(corrupted_images),
        "corrupted_samples": corrupted_images[:5],
    }


def inspect_path(target_path: Path) -> Dict[str, Any]:
    """Inspect directory or ZIP archive."""
    if target_path.is_file() and target_path.suffix.lower() == ".zip":
        return inspect_zip(target_path)
    elif target_path.is_dir():
        # Check if directory contains a ZIP archive
        zips = list(target_path.glob("*.zip"))
        if zips and not any(target_path.glob("**/*.jpg")) and not any(target_path.glob("**/*.png")):
            return inspect_zip(zips[0])
        return inspect_directory(target_path)
    else:
        raise ValueError(f"Target path must be a directory or a .zip file: {target_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="FarmWatch Dataset Inspection Utility")
    parser.add_argument(
        "--path",
        type=str,
        default="data/raw",
        help="Path to raw dataset directory or .zip archive (default: data/raw)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/reports/dataset_inspection.json",
        help="Path to save machine-readable inspection JSON report",
    )
    args = parser.parse_args()

    target_path = Path(args.path)
    if not target_path.exists():
        print(f"[ERROR] Path does not exist: {target_path}", file=sys.stderr)
        return 1

    print(f"[INFO] Inspecting dataset at: {target_path}")
    try:
        report = inspect_path(target_path)
    except Exception as e:
        print(f"[ERROR] Inspection failed: {e}", file=sys.stderr)
        return 1

    print("\n--- DATASET INSPECTION SUMMARY ---")
    print(f"Source Type: {report.get('source_type')}")
    print(f"Total Entries/Files: {report.get('total_entries') or report.get('total_files')}")
    if "file_size_bytes" in report:
        print(f"Archive Size: {report['file_size_bytes'] / (1024 * 1024):.2f} MB")
        print(f"MD5: {report.get('md5')}")
        print(f"SHA-256: {report.get('sha256')}")

    print("\nExtensions:")
    for ext, count in report.get("extensions", {}).items():
        print(f"  {ext or '<no-extension>'}: {count}")

    print("\nDetected Class Folders:")
    for sub, count in report.get("detected_subfolders", {}).items():
        print(f"  {sub}: {count} images")

    print("\nCanonical Class Matches:")
    for c, count in report.get("detected_classes", {}).items():
        print(f"  {c}: {count}")

    print("\nSample Colour Modes:", report.get("sample_modes"))
    print("Sample Dimensions:", report.get("sample_dimensions"))
    print("Metadata Files:", report.get("metadata_files"))
    print(f"Corrupted Images Detected: {report.get('corrupted_images_detected')}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[SUCCESS] Inspection report saved to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
