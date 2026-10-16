# FarmWatch — Field Crop Water Stress & Condition Classifier

> **AI-assisted visible condition classification for agricultural crop canopies using PyTorch transfer learning, FastAPI, and React.**

---

## 1. Overview: What FarmWatch Does & Does Not Do

### What FarmWatch Does
- **Visible Condition Estimation:** Classifies aerial or close-canopy photographs of maize (*Zea mays*) into five canonical condition classes:
  - `healthy`: Intact canopy with normal chlorophyll vigor and zero lesion development.
  - `low_stress`: Early or mild abiotic moisture stress (slight turgor loss, subtle upward leaf rolling).
  - `high_stress`: Pronounced acute drought stress (pencil leaf rolling, tip scorching, wilting).
  - `rust`: Common corn rust (*Puccinia sorghi*) foliar fungal infection with pustule uredinia.
  - `bare_soil`: Ground / soil background patches without crop foliage.
- **FastAPI Model Inference:** Serves deterministic, zero-gradient PyTorch inference (`torch.inference_mode()`) with strict input sanitization, EXIF transposition, and memory-safe bounded uploads.
- **Interactive Agronomic UI:** Provides an accessible, responsive dashboard with drag-and-drop file upload, held-out test sample loading, real-time backend health monitoring, confidence warning banners, and condition probability distributions.
- **Diagnostic Transparency:** Returns host crop identity, botanical pathogen names, visual diagnostic checklists, and cautious agricultural interpretations.

### What FarmWatch Does NOT Do
- **Not a Certified Agricultural Diagnosis:** FarmWatch provides **statistical machine-learning estimates**, not guaranteed laboratory pathology or agronomist certification.
- **Not Treatment or Agrochemical Prescription:** Does **not** prescribe fertilizers, irrigation volumes, fungicides, or pesticides. Always consult local extension services and certified crop advisors.
- **Not a Future Yield or Stress Forecast:** Classifies visible symptoms in the submitted snapshot; does not model weather forecasts or forecast future yield loss.
- **Not a Universal Crop Classifier:** Trained exclusively on maize canopy imagery from a specific UAV field flight; out-of-domain images (weeds, unrelated crops, livestock, blurred photos) are subject to closed-set classification limits.

---

## 2. Tech Stack & Architecture Overview

```
                      +---------------------------------------+
                      |             React + Vite              |
                      |   Frontend Dashboard (Port 5173)     |
                      +-------------------+-------------------+
                                          |
                                          | HTTP / REST (Multipart)
                                          v
                      +-------------------+-------------------+
                      |             FastAPI                   |
                      |    Backend Server (Port 8000)         |
                      +-------------------+-------------------+
                                          |
                                          | Lifespan Pre-loaded
                                          v
                      +-------------------+-------------------+
                      |      PyTorch ResNet-18 Model          |
                      |  ImageNet Weights + Custom Fine-tune  |
                      |  artifacts/models/farmwatch_resnet18  |
                      +---------------------------------------+
```

| Layer | Technologies |
| :--- | :--- |
| **Deep Learning** | PyTorch (`2.14.1+cpu`), Torchvision (`0.29.1+cpu`), ImageNet Pretrained ResNet-18 |
| **Data & Metrics** | Pillow (`12.3.0`), scikit-learn (`1.9.1`), NumPy (`2.4.6`), Matplotlib (`3.11.2`) |
| **Backend REST API** | FastAPI (`0.143.2`), Uvicorn (`0.54.0`), Pydantic Settings, `python-multipart` |
| **Frontend UI** | React 19, Vite 8, Vanilla CSS (Agricultural design tokens, zero Tailwind bloat) |
| **Testing & Quality** | Pytest (`9.1.1`), HTTPX (`0.28.1`), Ruff (`0.17.1`), Oxlint (`1.81.0`) |

---

## 3. Dataset Source, License & Acquisition

- **Source Record:** Zenodo Record 19108517 — *UAV-Based Multispectral Maize Dataset for Water Stress and Rust Detection*
- **DOI:** [10.5281/zenodo.19108517](https://doi.org/10.5281/zenodo.19108517)
- **License:** Creative Commons Attribution 4.0 International (CC-BY-4.0)
- **Target Subset:** 3-channel RGB subset (`mini_dataset_3ch`) containing 750 valid 128x128 TIFF images (150 images per class).
- **Leakage Prevention:** 14 exact SHA-256 duplicate clusters quarantined; deterministic stratified split into:
  - **Train:** 527 samples
  - **Validation:** 109 samples
  - **Held-out Test:** 114 samples

### Automated Dataset Preparation
```bash
# 1. Download archive via Zenodo API
python scripts/download_dataset.py

# 2. Inspect archive integrity, counts, and Pillow decoding
python scripts/inspect_dataset.py

# 3. Detect duplicate clusters and verify zero cross-split leakage
python scripts/verify_dataset.py

# 4. Generate splits manifest (data/processed/splits.csv)
python scripts/prepare_dataset.py
```

---

## 4. Prerequisites & Environment Setup

- **Python:** 3.11 recommended (`3.11.9` verified)
- **Node.js:** v20+ (`v24.21.0` verified)
- **npm:** 10+ (`11.17.0` verified)

### Python Virtual Environment Setup

**Windows PowerShell:**
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

**macOS / Linux:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Frontend Dependencies Setup
```bash
cd frontend
npm install
cd ..
```

---

## 5. Model Training & Evaluation

The baseline model is a fine-tuned ResNet-18 trained with AdamW, Cosine Annealing learning rate schedule, and validation Macro F1 early-checkpointing.

```bash
# Train ResNet-18 baseline
python -m training.train --epochs 10 --batch-size 16 --lr 1e-4

# Evaluate trained checkpoint on held-out test split (114 samples)
python -m training.evaluate --checkpoint artifacts/models/farmwatch_resnet18.pth
```

### Verified Benchmark Performance (Held-Out Test Set)
- **Test Accuracy:** **93.86%**
- **Macro Precision:** **0.9435**
- **Macro Recall:** **0.9391**
- **Macro F1 Score:** **0.9376**
- **Test Cross-Entropy Loss:** **0.1499**

Generated artifacts are stored in:
- Model Checkpoint: `artifacts/models/farmwatch_resnet18.pth`
- Per-class Report: `artifacts/reports/classification_report.csv`
- Confusion Matrix: `artifacts/reports/confusion_matrix.png`
- Training Curves: `artifacts/reports/training_curves.png`
- Evaluation Summary: `artifacts/reports/evaluation_results.json`

---

## 6. Running the Application Locally

Run the backend and frontend in separate terminals.

### Terminal A — FastAPI Backend
```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```
* Backend API available at: `http://127.0.0.1:8000`
* Interactive OpenAPI Swagger Docs: `http://127.0.0.1:8000/docs`

### Terminal B — React Frontend
```powershell
# Windows PowerShell
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173
```
* Web Dashboard available at: `http://127.0.0.1:5173`

---

## 7. API Specification

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Service root, API version, operational status, and link to docs |
| `GET` | `/health` | Live health probe, loaded model architecture, inference device, and class count |
| `GET` | `/api/classes` | Canonical class catalog with display names, pathogen taxonomy, and symptoms |
| `POST` | `/api/predict` | Classifies an uploaded crop photograph (`multipart/form-data`, key: `file`) |
| `GET` | `/api/sample/{class_name}` | Returns a sample test image converted to JPEG for rapid testing |

### Example Prediction Response (`POST /api/predict`)
```json
{
  "predicted_class": "rust",
  "display_name": "Common Corn Rust (Puccinia sorghi)",
  "category_type": "disease",
  "disease_name": "Common Corn Rust",
  "scientific_name": "Puccinia sorghi",
  "pathogen_type": "Fungal Pathogen (Basidiomycota)",
  "host_crop": "Maize / Corn (Zea mays)",
  "symptoms": "Small, circular to elongate golden-brown to cinnamon pustules (uredinia) scattered across both upper and lower leaf surfaces.",
  "confidence": 0.9999,
  "confidence_warning": false,
  "probabilities": {
    "bare_soil": 0.0,
    "healthy": 0.0,
    "high_stress": 0.0,
    "low_stress": 0.0,
    "rust": 0.9999
  },
  "interpretation": "The image resembles the 'Common Corn Rust (Puccinia sorghi)' class in the training dataset with 100.0% estimated confidence.",
  "model_version": "1.0.0",
  "notice": "AI image classification estimate based on maize training dataset. Not a confirmed agricultural diagnosis."
}
```

---

## 8. Automated Testing & Code Quality

Run tests and linters from the repository root:

```bash
# Run backend pytest suite (16 tests across API, dataset utils, and model utils)
python -m pytest

# Check Python code formatting with Ruff
ruff check .

# Check React frontend linting
cd frontend && npm run lint

# Verify frontend production build
npm run build
```

---

## 9. Configuration & Environment Variables

Create `.env` in the root directory for backend overrides:

```env
# Backend Server Configuration
HOST=127.0.0.1
PORT=8000
FARMWATCH_MODEL_PATH=artifacts/models/farmwatch_resnet18.pth
FRONTEND_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
MAX_UPLOAD_MB=10
CONFIDENCE_WARNING_THRESHOLD=0.60
```

Create `frontend/.env` for frontend overrides:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

---

## 10. Citations & Attribution

If using this codebase or dataset for academic research, please cite:

```bibtex
@dataset{zenodo_19108517,
  title     = {UAV-Based Multispectral Maize Dataset for Water Stress and Rust Detection},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19108517},
  url       = {https://doi.org/10.5281/zenodo.19108517}
}

@article{he2016deep,
  title     = {Deep Residual Learning for Image Recognition},
  author    = {He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  journal   = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2016}
}
```
