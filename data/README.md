# FarmWatch Data Documentation

## Provenance & Source
- **Dataset Title:** UAV-Based Multispectral Maize Dataset for Water Stress and Rust Detection (Water Stress 2025 and Common Rust 2025) (Subset)
- **Record Identifier:** Zenodo Record 19108517
- **DOI:** 10.5281/zenodo.19108517
- **Landing Page:** https://zenodo.org/records/19108517
- **API Endpoint:** https://zenodo.org/api/records/19108517
- **Access Date:** 2026-10-16
- **License:** Creative Commons Attribution 4.0 International (CC-BY-4.0)
- **Citation Instructions:** Cite Zenodo Record 10.5281/zenodo.19108517 according to source terms.

## Archive Specifications
- **Archive Filename:** `mini_datasets.zip`
- **Archive File Size:** 23,448,300 bytes (~22.36 MB)
- **MD5 Checksum:** `5c90bb1dab13780bfb0bacfcd9efc024` (Verified)
- **SHA-256 Checksum:** `522bf06571ad003b682727bc28304baf0ca0cfff8ff4d7b45c1e7f402003bed7`

## Extracted Layout & Target Subset
The archive extracts into `data/raw/extracted/mini_datasets/` with two subsets:
- `mini_dataset_6ch/`: 6-channel multispectral patches (not used for the 3-channel ResNet-18 MVP baseline).
- `mini_dataset_3ch/`: **Selected Target Subset**. 3-channel RGB patches in TIFF format (`.tif`).

### Image & Label Statistics (3-Channel Subset)
- **Resolution:** 128 x 128 pixels
- **Colour Mode:** RGB (3 channels)
- **Corrupted Images:** 0 / 750 (All images successfully verified and decoded with Pillow)
- **Per-Class Image Counts:**
  - `healthy`: 150 images
  - `low_stress`: 150 images
  - `high_stress`: 150 images
  - `rust`: 150 images
  - `bare_soil`: 150 images
- **Total Valid Images:** 750 images

## Preprocessing & Split Policy
- Images will be audited for exact SHA-256 duplicates across classes.
- Splits will be created deterministically (70% train / 15% val / 15% test) with fixed random seed (42).
- Split assignments will be tracked in `data/processed/splits.csv`.
