# FarmWatch Data Directory

This directory contains raw and processed datasets for FarmWatch.

## Directory Structure
- `raw/`: Unmodified downloaded archives and unzipped dataset files. (Ignored by Git)
- `processed/`: Formatted train/val/test splits and dataset split manifests (e.g., `splits.csv`). (Ignored by Git)

## Primary Dataset Source (Zenodo)
- **Record**: Zenodo Record 19108517
- **URL**: https://zenodo.org/records/19108517
- **API**: https://zenodo.org/api/records/19108517
- **Target Subset**: 3-channel RGB image patches
- **Canonical Labels**:
  1. `healthy`
  2. `low_stress`
  3. `high_stress`
  4. `rust`
  5. `bare_soil`
