# FarmWatch — Implementation Status

## Phase Checklist

- [x] **Phase 0 — Inspect workspace before editing**
  - Workspace inspected: clean git repo (`main` branch), untracked `steps.md`.
  - Python versions detected: 3.13.1 (default) and 3.11 (`py -3.11`). Selected Python 3.11 for PyTorch ecosystem compatibility.
  - Node & npm verified: Node v24.21.0, npm 11.17.0.
  - Project directory structure initialized.
  - Status tracker and initial work log created.

- [x] **Phase 1 — Python/Node setup and dependency files**
  - [x] Virtual environment `.venv` created with Python 3.11 (`3.11.9`)
  - [x] `requirements.txt` and `requirements-dev.txt` created and dependencies installed
  - [x] PyTorch + Torchvision verified (`torch 2.14.1+cpu`, `torchvision 0.29.1+cpu`, CUDA available: False)
  - [x] Core dependencies verified (`fastapi 0.143.2`, `pillow 12.3.0`, `scikit-learn 1.9.1`, `numpy 2.4.6`, `matplotlib 3.11.2`)
  - [x] Dev dependencies verified (`pytest 9.1.1`, `ruff 0.17.1`)
  - [x] React frontend initialized with Vite, packages installed, build tested (`npm run build` passed)
  - [x] `.gitignore`, root `.env.example`, and `frontend/.env.example` created

- [x] **Phase 2 — Inspect and acquire dataset**
  - [x] Zenodo metadata inspected via API/landing page (Zenodo Record 19108517, CC-BY-4.0)
  - [x] Archive downloaded & verified: `mini_datasets.zip` (23.45 MB, MD5: `5c90bb1dab13780bfb0bacfcd9efc024`, SHA-256: `522bf06571ad003b682727bc28304baf0ca0cfff8ff4d7b45c1e7f402003bed7`)
  - [x] Provenance, licence, citation, and file statistics recorded in `data/README.md` and `docs/DATASET_CARD.md`
  - [x] CLI inspection script `scripts/inspect_dataset.py` implemented and verified
  - [x] Target 3-channel RGB subset (`mini_dataset_3ch`) verified: 750 valid 128x128 TIFF images (150 per class, 0 corrupted)
  - [x] Inspection report exported to `artifacts/reports/dataset_inspection.json`

- [x] **Phase 3 — Dataset verification, deduplication, and splits**
  - [x] Image integrity verified with Pillow (750 / 750 valid, 0 corrupt files)
  - [x] Duplicate detection performed: 14 exact SHA-256 duplicate clusters detected and quarantined to prevent cross-split leakage
  - [x] CLI script `scripts/verify_dataset.py` implemented and verified; report written to `artifacts/reports/dataset_verification.json`
  - [x] CLI script `scripts/prepare_dataset.py` implemented; generated deterministic split manifest `data/processed/splits.csv`
  - [x] Materialized `train` (527), `val` (109), and `test` (114) splits
  - [x] Zero duplicate cross-split leakage verified programmatically

- [x] **Phase 4 — Implement and train CNN baseline (ResNet18)**
  - [x] Model baseline + transfer learning in `training/model_utils.py`
  - [x] Training script `training/train.py` with validation macro F1 checkpointing
  - [x] Best checkpoint saved to `artifacts/models/farmwatch_resnet18.pth` (Val Macro F1: 0.9460)
  - [x] Training history (`artifacts/reports/training_history.csv`) and curves (`artifacts/reports/training_curves.png`) exported

- [x] **Phase 5 — Evaluate trained checkpoint**
  - [x] `training/evaluate.py` executed on held-out test split (114 samples)
  - [x] Final Test Metrics: Accuracy 93.86%, Macro Precision 0.9435, Macro Recall 0.9391, Macro F1 0.9376, Test Loss 0.1499
  - [x] Per-class report (`artifacts/reports/classification_report.csv`), confusion matrix (`artifacts/reports/confusion_matrix.png`), and results JSON (`artifacts/reports/evaluation_results.json`) exported

- [x] **Phase 6 — Inference service (FastAPI backend)**
  - [x] `backend/app/model_service.py` with strict checkpoint loader, lifespan initialization, and deterministic preprocessing
  - [x] Schemas, config, error handling, CORS in `backend/app/`
  - [x] Endpoints verified: `GET /`, `GET /health`, `GET /api/classes`, `POST /api/predict`

- [x] **Phase 7 — React frontend**
  - [x] Clean, responsive UI with deep green/sage agricultural theme in `frontend/src/`
  - [x] Components implemented: `ImageUploader.jsx`, `PredictionResult.jsx`, `ProbabilityList.jsx`, `api.js`
  - [x] Production build verified (`npm run build`) and lint verified (`npm run lint` / oxlint: 0 warnings, 0 errors)

- [x] **Phase 8 — End-to-end integration**
  - [x] FastAPI backend inference pipeline integrated with React frontend contract
  - [x] Safe multipart file handling, corrupted file rejection (400), unsupported MIME validation, and missing file handling (422) verified

- [x] **Phase 9 — Automated tests and code quality**
  - [x] Full pytest test suite passing: 16 passed in 12.57s (`tests/test_api.py`, `tests/test_dataset_utils.py`, `tests/test_model_utils.py`)
  - [x] Frontend build & lint verified with 0 errors

- [x] **Phase 10 — Documentation and reproducibility**
  - [x] Comprehensive root `README.md` created with architecture, setup, API contracts, benchmark metrics, and citations
  - [x] `docs/MODEL_CARD.md`, `docs/DATASET_CARD.md`, `docs/DECISIONS.md` created and maintained
  - [x] `docs/IMPLEMENTATION_STATUS.md` maintained

- [x] **Phase 11 — Deployment readiness**
  - [x] Production build and hosting deployment guide in `docs/DEPLOYMENT.md`
  - [x] Hardened multi-stage non-root `Dockerfile` created
  - [x] Production `docker-compose.yml` created with backend healthcheck and frontend service wiring
  - [x] CORS origin controls, payload bounding, and memory protection verified

- [x] **Phase 12 — Optional enhancements**
  - [x] Grad-CAM visual attribution module implemented (`training/gradcam.py`)
  - [x] Grad-CAM API integration in `backend/app/schemas.py` and `backend/app/model_service.py`
  - [x] Interactive UI attention heatmap viewer in `frontend/src/components/PredictionResult.jsx` and `App.css`
  - [x] Model architecture comparison benchmark utility implemented (`training/compare_models.py`)
  - [x] Comparative evaluation executed between ResNet-18 (96.49% test acc, 20.88ms CPU latency) and MobileNetV3-Small (4.84ms latency, 1.5M params); exported to `artifacts/reports/model_comparison.json` and `.csv`
- [x] **Phase 13: Multi-Crop Specialist Expansion (Potato & Tomato / Solanaceae)**
  - [x] High-quality dataset acquired from PlantVillage (9 classes: Potato Early Blight, Late Blight, Healthy; Tomato Early Blight, Late Blight, Bacterial Spot, Yellow Leaf Curl, Healthy; Bare Soil controls)
  - [x] Automated dataset download & stratified split generator (`scripts/prepare_solanaceae_dataset.py`, 360 images)
  - [x] Transfer learning pipeline (`training/train_solanaceae.py`) with 2-stage fine-tuning (Validation Macro F1: 0.9814, Test Accuracy: 85.19%)
  - [x] Modular Multi-Model Backend Service (`backend/app/model_service.py`) supporting dynamic crop routing (`crop=maize|solanaceae`)
  - [x] Comprehensive pathogen & agronomic metadata for all Solanaceae classes (`SOLANACEAE_METADATA`)
  - [x] Frontend Crop Switcher tab bar (`frontend/src/App.jsx`, `App.css`) with dynamic sample buttons
  - [x] Side-by-side visual comparison pane (Original Photo & AI Attention Heatmap Grad-CAM)
  - [x] Automated multi-crop regression and integration test suite (`tests/test_solanaceae.py`, 20 tests total passing)

---

## Internal Work Log

- **2026-10-16**:
  - Checked working directory `e:\GenAI_Hack` and `d:\GenAI_Hack`.
  - Git repository on branch `main` with `steps.md`.
  - Detected Node `v24.21.0` and npm `11.17.0`.
  - Detected Python `3.11.9`. Selected Python 3.11 for PyTorch ecosystem compatibility.
  - Completed Phase 0 through Phase 9 acceptance checks.
  - Implemented Phase 10: Created comprehensive root `README.md` covering all 14 execution guidelines.
  - Implemented Phase 11: Created `docs/DEPLOYMENT.md`, multi-stage `Dockerfile`, and `docker-compose.yml`.
  - Implemented Phase 12 Enhancements:
    - Built Grad-CAM attribution generator (`training/gradcam.py`), hooked into FastAPI inference, and added UI heatmap viewer card.
    - Built model comparison benchmarking script (`training/compare_models.py`) and generated comparative latency/accuracy metrics between ResNet-18 and MobileNetV3-Small.
  - Implemented Phase 13 Multi-Crop Specialist (Potato & Tomato):
    - Acquired and balanced 9-class Solanaceae dataset (`data/processed_solanaceae/splits.csv`).
    - Trained specialist ResNet-18 model (`artifacts/models/farmwatch_solanaceae_resnet18.pth`).
    - Enhanced FastAPI `ModelService` with modular multi-model loader and automated specialist routing (`crop=auto`).
    - Streamlined UX: Removed tab switches in favor of a single unified upload flow with auto-crop detection and a curated strip of supported demo samples.
    - Verified full test suite: 20 passed in 8.49s (`tests/test_api.py`, `tests/test_dataset_utils.py`, `tests/test_model_utils.py`, `tests/test_solanaceae.py`).
  - Verified frontend code quality: Oxlint 0 warnings / 0 errors, Vite production build clean (119ms).
  - All project phases and verification criteria fully completed.
