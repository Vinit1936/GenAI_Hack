# Dataset Card: Zenodo Record 19108517 (Maize Water Stress & Condition RGB Subset)

## 1. Summary
- **Title:** UAV-Based Multispectral Maize Dataset for Water Stress and Rust Detection (Water Stress 2025 and Common Rust 2025) (Subset)
- **Record Identifier:** Zenodo Record 19108517
- **DOI:** 10.5281/zenodo.19108517
- **Landing Page:** https://zenodo.org/records/19108517
- **API URL:** https://zenodo.org/api/records/19108517
- **License:** CC-BY-4.0
- **Acquisition Date:** 2026-10-16
- **Archive File:** `mini_datasets.zip` (23,448,300 bytes)
- **MD5:** `5c90bb1dab13780bfb0bacfcd9efc024`
- **SHA-256:** `522bf06571ad003b682727bc28304baf0ca0cfff8ff4d7b45c1e7f402003bed7`

## 2. Taxonomy & Classes
The target RGB subset (`mini_dataset_3ch`) consists of 750 image patches across 5 canonical classes:
1. `healthy` (150 images): Maize canopy showing normal pigmentation and vigor without severe drought symptoms.
2. `low_stress` (150 images): Early/mild water stress symptoms.
3. `high_stress` (150 images): Severe water stress, pronounced wilting and leaf curling.
4. `rust` (150 images): Maize foliage exhibiting visible fungal rust pustules.
5. `bare_soil` (150 images): Soil background patches without crop foliage.

## 3. Image Characteristics
- **Raw Format:** TIFF (`.tif`), 3-channel RGB.
- **Native Dimensions:** 128 x 128 pixels.
- **Verification:** 100% (750/750) decoded successfully with Pillow; 0 corrupt files.
- **Model Input Resizing:** Bilinear interpolation to 224 x 224 pixels to match ImageNet standard.
- **Normalization:** ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`.

## 4. Split and Leakage Control
- **Splits:** 70% Train (~105 per class, 525 total) / 15% Validation (~22-23 per class, 112 total) / 15% Test (~22-23 per class, 113 total).
- **Seed:** Fixed seed `42`.
- **Auditing:** SHA-256 hash calculated for all 750 images; exact duplicates checked before partitioning.
- **Manifest:** Saved to `data/processed/splits.csv`.

## 5. Known Limitations
- The mini subset contains 150 sampled patches per class (750 total). While sufficient for fine-tuning a pre-trained feature extractor, confidence estimates carry statistical variance on small test subsets (~113 test samples).
- UAV aerial perspective differs from smartphone close-up angles.
- The classifier is closed-set: non-maize imagery will still receive one of the 5 labels.
