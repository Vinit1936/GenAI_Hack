"""Tests for FarmWatch dataset inspection, verification, and split utilities."""

from __future__ import annotations

import csv
import io
import tempfile
from pathlib import Path
from PIL import Image
import pytest

from scripts.inspect_dataset import is_safe_zip_path, inspect_image_bytes
from scripts.verify_dataset import calculate_sha256, calculate_dhash, verify_dataset
from scripts.prepare_dataset import prepare_dataset, CANONICAL_CLASSES


def test_safe_zip_path():
    assert is_safe_zip_path("images/healthy_01.tif") is True
    assert is_safe_zip_path("../secret.txt") is False
    assert is_safe_zip_path("foo/../../bar.jpg") is False


def test_inspect_image_bytes_valid():
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    valid, mode, size, err = inspect_image_bytes(buf.getvalue(), "test.jpg")
    assert valid is True
    assert mode == "RGB"
    assert size == (64, 64)
    assert err is None


def test_inspect_image_bytes_corrupt():
    corrupt_bytes = b"NOT_AN_IMAGE_FILE_DATA_12345"
    valid, mode, size, err = inspect_image_bytes(corrupt_bytes, "bad.jpg")
    assert valid is False
    assert err is not None


def test_dhash_consistency():
    img1 = Image.new("RGB", (32, 32), color=(50, 50, 50))
    img2 = Image.new("RGB", (32, 32), color=(50, 50, 50))
    assert calculate_dhash(img1) == calculate_dhash(img2)


def test_deterministic_split_and_no_leakage(tmp_path):
    # Create synthetic dataset with known duplicates in one class
    for cls in CANONICAL_CLASSES:
        cls_dir = tmp_path / "raw" / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        # Create 10 unique images
        for i in range(10):
            img = Image.new("RGB", (16, 16), color=(i * 20, 100, 50))
            img.save(cls_dir / f"img_{i:02d}.png")

    # Add duplicate file with identical bytes
    shutil_copy = tmp_path / "raw" / "healthy" / "img_dup.png"
    with open(tmp_path / "raw" / "healthy" / "img_00.png", "rb") as f_src:
        with open(shutil_copy, "wb") as f_dst:
            f_dst.write(f_src.read())

    out_dir = tmp_path / "processed"
    manifest = prepare_dataset(
        source_dir=tmp_path / "raw",
        output_dir=out_dir,
        seed=42,
        materialize_folders=True,
    )

    assert manifest.is_file()

    # Verify duplicate images are assigned to the same split
    hash_splits = {}
    with open(manifest, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            sha = r["sha256"]
            split = r["split"]
            if sha in hash_splits:
                assert hash_splits[sha] == split, f"Leak detected for SHA {sha}"
            else:
                hash_splits[sha] = split
