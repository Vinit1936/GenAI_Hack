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

- [ ] **Phase 4 — Implement and train CNN baseline (ResNet18)**
  - [ ] Model baseline + transfer learning in `training/model_utils.py`
  - [ ] Training script `training/train.py` with validation macro F1 checkpointing
  - [ ] Training history and curves exported

- [ ] **Phase 5 — Evaluate trained checkpoint**
  - [ ] `training/evaluate.py` with held-out test evaluation
  - [ ] Macro F1, confusion matrix, per-class metrics report exported

- [ ] **Phase 6 — Inference service (FastAPI backend)**
  - [ ] `backend/app/model_service.py` with strict checkpoint loader and preprocessing
  - [ ] Schemas, config, error handling, CORS in `backend/app/`
  - [ ] Endpoints: `/health`, `/api/classes`, `/api/predict`

- [ ] **Phase 7 — React frontend**
  - [ ] Clean, responsive UI with deep green/sage theme
  - [ ] Drag-and-drop uploader, image preview, reset action
  - [ ] Prediction card, probability distribution, confidence/disclaimer banners
  - [ ] Integration with backend API

- [ ] **Phase 8 — End-to-end integration**
  - [ ] Live communication between Vite frontend and FastAPI backend verified
  - [ ] Error cases, validation, edge cases tested

- [ ] **Phase 9 — Automated tests and code quality**
  - [ ] Pytest suite for dataset utils, model loading/inference, and API routes
  - [ ] Frontend build verification (`npm run build`)

- [ ] **Phase 10 — Documentation and reproducibility**
  - [ ] Comprehensive `README.md`, `docs/MODEL_CARD.md`, `docs/DATASET_CARD.md`, `docs/DECISIONS.md`

- [ ] **Phase 11 — Deployment readiness**
  - [ ] Frontend production build & backend deployment configurations documented

- [ ] **Phase 12 — Optional enhancements**
  - [ ] Grad-CAM visualization
  - [ ] Model comparisons (MobileNetV3 / EfficientNet-B0)

---

## Internal Work Log

- **2026-10-16**:
  - Checked working directory `e:\GenAI_Hack`.
  - Git repository on branch `main` with `steps.md`.
  - Detected Node `v24.15.0` and npm `11.17.0`.
  - Detected Python `3.13.1` and `3.11.9`. Selected Python 3.11 (`py -3.11`) for PyTorch compatibility.
  - Completed Phase 0 acceptance checks.
  - Initialized `.venv` using Python 3.11 (`3.11.9`).
  - Installed PyTorch (CPU `2.14.1+cpu`), Torchvision (`0.29.1+cpu`), FastAPI (`0.143.3`), Uvicorn (`0.54.0`), Pillow (`12.3.0`), scikit-learn (`1.9.1`), Matplotlib (`3.11.2`), Pydantic (`2.14.1`), pytest (`9.1.1`), ruff (`0.17.1`).
  - Verified Python imports: `torch 2.14.1+cpu`, `torchvision 0.29.1+cpu`, `CUDA: False`, `fastapi`, `PIL`, `sklearn`, `numpy`, `matplotlib`.
  - Installed npm packages in `frontend/`, verified production build (`npm run build`) and linting (`npm run lint` with oxlint).
  - Verified `.gitignore`, root `.env.example`, and `frontend/.env.example`.
  - Completed Phase 1 acceptance checks completely.
