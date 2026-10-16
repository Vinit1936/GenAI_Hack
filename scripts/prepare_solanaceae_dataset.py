"""Dataset Acquisition and Preparation for Potato and Tomato (Solanaceae) Specialist.

Acquires a balanced, verified subset of Potato and Tomato foliar conditions from
the open PlantVillage repository (spMohanty/PlantVillage-Dataset) combined with
bare soil control samples from the field dataset.
Generates deterministic stratified splits and materializes:
  data/processed_solanaceae/train/
  data/processed_solanaceae/val/
  data/processed_solanaceae/test/
  data/processed_solanaceae/splits.csv
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
from pathlib import Path
import random
import shutil
import time
from typing import Dict, List, Tuple

from PIL import Image
import requests

GITHUB_API_BASE = "https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color"

CLASS_FOLDER_MAPPING = {
    "potato_early_blight": "Potato___Early_blight",
    "potato_late_blight": "Potato___Late_blight",
    "potato_healthy": "Potato___healthy",
    "tomato_bacterial_spot": "Tomato___Bacterial_spot",
    "tomato_early_blight": "Tomato___Early_blight",
    "tomato_late_blight": "Tomato___Late_blight",
    "tomato_yellow_leaf_curl": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "tomato_healthy": "Tomato___healthy",
}

SAMPLES_PER_CLASS = 40
RANDOM_SEED = 42


def download_class_samples(
    canonical_class: str,
    github_folder: str,
    target_dir: Path,
    limit: int = SAMPLES_PER_CLASS,
) -> List[Tuple[Path, str]]:
    """Downloads sample images for a given class from GitHub repository."""
    target_dir.mkdir(parents=True, exist_ok=True)
    api_url = f"{GITHUB_API_BASE}/{github_folder}"
    print(f"[INFO] Fetching file catalog for '{canonical_class}' from {github_folder}...")

    headers = {"User-Agent": "FarmWatch-Dataset-Builder"}
    res = requests.get(api_url, headers=headers, timeout=15)
    if res.status_code != 200:
        raise RuntimeError(f"Failed to fetch catalog for {github_folder}: HTTP {res.status_code}")

    items = [item for item in res.json() if item["name"].lower().endswith((".jpg", ".jpeg", ".png"))]
    # Deterministic selection
    rng = random.Random(RANDOM_SEED)
    selected_items = rng.sample(items, min(limit, len(items)))

    saved_records = []
    for idx, item in enumerate(selected_items):
        file_name = f"{canonical_class}_{idx+1:03d}.jpg"
        dest_path = target_dir / file_name

        if not dest_path.is_file():
            download_url = item["download_url"]
            img_res = requests.get(download_url, headers=headers, timeout=15)
            if img_res.status_code != 200:
                print(f"[WARN] Failed to download {download_url}, skipping.")
                continue

            with open(dest_path, "wb") as f:
                f.write(img_res.content)

        # Compute SHA-256
        with open(dest_path, "rb") as f:
            sha256 = hashlib.sha256(f.read()).hexdigest()

        saved_records.append((dest_path, sha256))

    print(f"[INFO] Successfully acquired {len(saved_records)} verified samples for '{canonical_class}'.")
    return saved_records


def sample_soil_backgrounds(
    source_soil_dir: Path,
    target_dir: Path,
    limit: int = SAMPLES_PER_CLASS,
) -> List[Tuple[Path, str]]:
    """Copies bare soil samples as non-crop background control class."""
    target_dir.mkdir(parents=True, exist_ok=True)
    soil_files = sorted(list(source_soil_dir.glob("*.tif")))
    rng = random.Random(RANDOM_SEED)
    selected = rng.sample(soil_files, min(limit, len(soil_files)))

    saved_records = []
    for idx, src_path in enumerate(selected):
        dest_name = f"bare_soil_{idx+1:03d}.jpg"
        dest_path = target_dir / dest_name

        if not dest_path.is_file():
            # Convert TIFF to JPEG for unified processing
            with Image.open(src_path) as img:
                img.convert("RGB").save(dest_path, format="JPEG", quality=95)

        with open(dest_path, "rb") as f:
            sha256 = hashlib.sha256(f.read()).hexdigest()

        saved_records.append((dest_path, sha256))

    print(f"[INFO] Prepared {len(saved_records)} bare soil control samples.")
    return saved_records


def prepare_dataset(
    output_base: Path = Path("data/processed_solanaceae"),
    source_soil_dir: Path = Path("data/raw/extracted/mini_datasets/mini_dataset_3ch/bare_soil"),
) -> None:
    """End-to-end dataset acquisition, verification, and train/val/test splitting."""
    raw_cache = Path("data/raw_solanaceae")
    raw_cache.mkdir(parents=True, exist_ok=True)

    all_class_records: Dict[str, List[Tuple[Path, str]]] = {}

    # 1. Download Potato and Tomato classes
    for canonical_name, folder_name in CLASS_FOLDER_MAPPING.items():
        records = download_class_samples(
            canonical_class=canonical_name,
            github_folder=folder_name,
            target_dir=raw_cache / canonical_name,
            limit=SAMPLES_PER_CLASS,
        )
        all_class_records[canonical_name] = records
        time.sleep(0.5)  # Rate limiting courteous pause

    # 2. Add Bare Soil background control
    soil_records = sample_soil_backgrounds(
        source_soil_dir=source_soil_dir,
        target_dir=raw_cache / "bare_soil",
        limit=SAMPLES_PER_CLASS,
    )
    all_class_records["bare_soil"] = soil_records

    # 3. Stratified deterministic splitting (70% train / 15% val / 15% test)
    rng = random.Random(RANDOM_SEED)
    manifest_rows = []

    train_dir = output_base / "train"
    val_dir = output_base / "val"
    test_dir = output_base / "test"

    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    for class_name, items in all_class_records.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)

        train_items = items[:n_train]
        val_items = items[n_train : n_train + n_val]
        test_items = items[n_train + n_val :]

        for split_name, split_items in [("train", train_items), ("val", val_items), ("test", test_items)]:
            split_class_dir = output_base / split_name / class_name
            split_class_dir.mkdir(parents=True, exist_ok=True)

            for src_path, sha in split_items:
                dest_path = split_class_dir / src_path.name
                shutil.copy2(src_path, dest_path)

                rel_path = f"{split_name}/{class_name}/{src_path.name}"
                manifest_rows.append({
                    "relative_path": rel_path,
                    "canonical_label": class_name,
                    "split": split_name,
                    "sha256": sha,
                })

    # 4. Save splits manifest
    splits_csv = output_base / "splits.csv"
    with open(splits_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["relative_path", "canonical_label", "split", "sha256"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    total_samples = len(manifest_rows)
    print(f"\n[SUCCESS] Solanaceae dataset prepared at '{output_base}':")
    print(f"  - Total samples: {total_samples}")
    print(f"  - Train: {len([r for r in manifest_rows if r['split'] == 'train'])}")
    print(f"  - Val:   {len([r for r in manifest_rows if r['split'] == 'val'])}")
    print(f"  - Test:  {len([r for r in manifest_rows if r['split'] == 'test'])}")
    print(f"  - Classes (9): {sorted(list(all_class_records.keys()))}")


if __name__ == "__main__":
    prepare_dataset()
