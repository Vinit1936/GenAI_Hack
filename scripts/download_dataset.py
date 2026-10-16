"""FarmWatch Dataset Download & Extraction Utility.

Downloads mini_datasets.zip from Zenodo (Record 19108517),
verifies its checksum, computes SHA-256, and extracts safely.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path
import requests
from tqdm import tqdm

ZENODO_RECORD_API = "https://zenodo.org/api/records/19108517"
DEFAULT_FILE_KEY = "mini_datasets.zip"
EXPECTED_MD5 = "5c90bb1dab13780bfb0bacfcd9efc024"


def calculate_hash(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "FarmWatch-Research/1.0"}
    response = requests.get(url, stream=True, headers=headers, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    print(f"[INFO] Downloading {dest_path.name} ({total_size / (1024*1024):.2f} MB)...")

    with open(dest_path, "wb") as f, tqdm(
        total=total_size, unit="B", unit_scale=True, desc=dest_path.name
    ) as bar:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))


def extract_archive(zip_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Safely extracting {zip_path.name} to {extract_dir}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.infolist():
            target_path = Path(member.filename)
            # Guard against path traversal
            if target_path.is_absolute() or ".." in target_path.parts:
                print(f"[WARN] Skipping unsafe entry: {member.filename}")
                continue
            z.extract(member, extract_dir)
    print("[SUCCESS] Extraction completed successfully.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download & extract FarmWatch Zenodo dataset")
    parser.add_argument("--dest", type=str, default="data/raw", help="Target raw data directory")
    parser.add_argument("--skip-extract", action="store_true", help="Do not extract the archive after download")
    args = parser.parse_args()

    dest_dir = Path(args.dest)
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_dest = dest_dir / DEFAULT_FILE_KEY

    # 1. Query Zenodo API to confirm file presence & link
    print(f"[INFO] Querying Zenodo Record API: {ZENODO_RECORD_API}")
    try:
        r = requests.get(ZENODO_RECORD_API, timeout=20)
        r.raise_for_status()
        record_data = r.json()
    except Exception as e:
        print(f"[ERROR] Failed to query Zenodo API: {e}", file=sys.stderr)
        return 1

    download_link = None
    expected_md5 = EXPECTED_MD5
    for f in record_data.get("files", []):
        if f.get("key") == DEFAULT_FILE_KEY:
            download_link = f.get("links", {}).get("self")
            checksum = f.get("checksum", "")
            if checksum.startswith("md5:"):
                expected_md5 = checksum[4:]
            break

    if not download_link:
        print(f"[ERROR] Could not find {DEFAULT_FILE_KEY} in Zenodo record.", file=sys.stderr)
        return 1

    # 2. Download if not already downloaded or checksum mismatch
    if zip_dest.is_file():
        print(f"[INFO] File already exists at {zip_dest}. Checking MD5...")
        current_md5 = calculate_hash(zip_dest, "md5")
        if current_md5 == expected_md5:
            print("[INFO] Existing archive matches expected MD5 checksum.")
        else:
            print("[WARN] Existing archive MD5 mismatch. Re-downloading...")
            download_file(download_link, zip_dest)
    else:
        download_file(download_link, zip_dest)

    # 3. Checksum verification
    actual_md5 = calculate_hash(zip_dest, "md5")
    actual_sha256 = calculate_hash(zip_dest, "sha256")
    print(f"[INFO] Verifying checksums:")
    print(f"  Expected MD5: {expected_md5}")
    print(f"  Actual MD5:   {actual_md5}")
    print(f"  SHA-256:      {actual_sha256}")

    if actual_md5 != expected_md5:
        print(f"[ERROR] Checksum verification failed! Corrupted download.", file=sys.stderr)
        return 1
    print("[SUCCESS] MD5 checksum verified.")

    # 4. Extract
    if not args.skip_extract:
        extract_dir = dest_dir / "extracted"
        extract_archive(zip_dest, extract_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
