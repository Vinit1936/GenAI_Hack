# FarmWatch — Detailed End-to-End Implementation Steps

> **Purpose of this file:** This is the execution specification for Antigravity (or another coding agent) to build FarmWatch from an empty or existing workspace. Execute the work, create the actual files, run the commands and tests that are possible, fix failures, and leave the repository in a working state. Do not stop after producing another plan.

## 0. Instructions to the coding agent

Act as the implementation engineer for **FarmWatch**, an image-based crop-condition classifier built with **PyTorch + transfer learning**, a **Python FastAPI backend**, and a **React + Vite frontend**.

### How to work

1. Inspect the current workspace before changing anything. Preserve existing work if the folder is not empty. If a project already exists, inspect its package files, source files, Git status, and README before making edits.
2. Implement the MVP in the phase order below. After every phase, run the relevant checks and fix the issues before moving on.
3. Do not merely print code snippets in chat. Create/edit the real project files in the workspace.
4. Do not stop after writing a plan. Continue implementing until the available work is complete. If an external dependency, download, GPU, or network connection is unavailable, implement everything that can be completed locally and document the exact blocker and manual action needed.
5. Do not ask the user questions for ordinary engineering choices. Use the sensible defaults in this document and record any necessary assumption in `README.md` or `docs/DECISIONS.md`.
6. Do not fabricate a trained model, dataset labels, validation metrics, accuracy, citations, or successful test results. A random/untrained model may be used only in isolated automated tests and must never be represented in the UI as a real prediction model.
7. Do not copy another GitHub project's implementation. Use the public datasets and model libraries as dependencies/data sources, and implement FarmWatch's own pipeline and interface.
8. Keep secrets, local environment files, data archives, generated splits, cache files, and large model files out of Git unless intentionally configured otherwise. Never put API keys into React; this project does not need API keys for basic predictions.
9. Maintain a small progress checklist in `docs/IMPLEMENTATION_STATUS.md` and update it as phases are completed. Keep `README.md` accurate.
10. Prefer simple, maintainable code. Add type hints to Python functions, meaningful error messages, comments where they clarify non-obvious logic, and accessible labels in React. Avoid unnecessary packages and architecture.
11. When blocked, don't invent a substitute silently. Continue with a clear data-free smoke test where appropriate, label it as a smoke test, and report what remains blocked.

### Required implementation result

A local developer must be able to:

1. Set up the Python environment.
2. download or manually place the selected dataset;
3. inspect the downloaded files and labels;
4. prepare deterministic train/validation/test splits without knowingly leaking near-duplicate samples;
5. train a pretrained CNN and save a checkpoint;
6. evaluate that checkpoint and save metrics/plots;
7. start the FastAPI server;
8. start the React frontend;
9. upload a photograph from React and receive a real prediction from the saved checkpoint;
10. run automated tests and follow the documented deployment instructions.

---

## 1. Product definition and guardrails

### 1.1 MVP goal

FarmWatch lets a user upload a crop/field image and receive a model prediction for one of the dataset's known visible-condition categories. It must clearly state that its result is an **AI image classification estimate**, not a confirmed agricultural diagnosis.

### 1.2 Default dataset and classes

Start with the **three-channel RGB subset** from the public Zenodo record:

- Dataset landing page: <https://zenodo.org/records/19108517>
- Dataset API record: <https://zenodo.org/api/records/19108517>
- The record describes RGB and multispectral subsets and up to 100 sampled image patches per class.
- The five labels described by the record are `low_stress`, `high_stress`, `rust`, `healthy`, and `bare_soil`.

Do **not** assume the ZIP's exact filename, archive structure, casing, image extensions, metadata columns, or file size. Inspect the record metadata and archive before implementing extraction or splitting. Dataset releases may change. Prefer a versioned, documented download and record the source URL, access date, licence, archive filename, and checksum in `data/README.md`.

Optional larger follow-up dataset:

- Full UAV multispectral maize dataset: <https://zenodo.org/records/20332029>
- It is not the starting point for the MVP. It is larger and may contain multispectral source imagery, patches, masks, and metadata. Do not download the full multi-gigabyte record without checking its file list, local disk space, and the user's likely need. A future segmentation/mapping phase may use it.

Alternative dataset for a separate disease-classification experiment only:

- PlantDoc: <https://github.com/pratikkayal/PlantDoc-Dataset>
- PlantDoc is a plant-disease dataset; do not pretend its labels are water-stress labels. Do not mix it into the default FarmWatch stress dataset without a carefully defined task and a documented class mapping.

### 1.3 Semantics of the five classes

The default RGB subset mixes stress, disease, vegetation, and background categories. Keep the original labels intact and describe them accurately in the product:

| Dataset label | Display label | Category type | Meaning for this MVP |
|---|---|---|---|
| `healthy` | Healthy-looking maize | `plant_condition` | Image resembles the dataset's healthy class; not proof of no stress. |
| `low_stress` | Low water-stress class | `water_stress` | Image resembles the low-stress class in the dataset. |
| `high_stress` | High water-stress class | `water_stress` | Image resembles the high-stress class in the dataset. |
| `rust` | Rust-disease class | `disease` | A separate disease class, not a stress-severity level. |
| `bare_soil` | Bare-soil/background class | `background` | The patch resembles the bare-soil class; it is not a crop diagnosis. |

The UI must call these **dataset categories / predicted visible condition**, not claim that all five are stress-severity levels. Do not rename a class in the model output unless an explicit class mapping is saved in the checkpoint/config.

### 1.4 Do not make unsupported claims

- The MVP classifies the provided image. It does not forecast future stress.
- A high softmax score is not necessarily a calibrated probability of real-world correctness.
- The first subset is small and comes from a particular maize/UAV data domain. Do not claim it is validated for all crops, smartphone images, farms, weather, geographies, or field conditions.
- Do not recommend pesticides, fertilizers, irrigation quantities, or other treatment actions based only on this classifier. The UI may recommend re-taking a clear image and consulting local agricultural guidance/experts.
- No computer-vision model should make a definite diagnosis when the uploaded image is poor, unrelated, or unlike its training data. Show confidence cautiously and communicate limitations.

---

## 2. Technology stack and architectural decisions

Use these defaults unless the workspace already has a working equivalent:

- **Python:** 3.11 or 3.12.
- **Deep learning:** PyTorch and Torchvision.
- **Model:** ImageNet-pretrained ResNet18 for the first baseline.
- **Image I/O:** Pillow.
- **Training/evaluation:** scikit-learn, NumPy, Matplotlib.
- **API:** FastAPI, Uvicorn, `python-multipart`.
- **API validation/tests:** Pydantic via FastAPI, pytest, HTTPX.
- **Frontend:** React, Vite, JavaScript (not TypeScript unless the workspace already uses TypeScript).
- **Frontend tests:** at minimum lint/build checks; add Vitest/React Testing Library if practical.
- **Source control:** Git.

### 2.1 Architecture

**Training pipeline (offline):**

`public dataset → inspect → validate labels/files → deduplicate/group split → transforms → pretrained ResNet18 → validation-based checkpoint selection → final test report → checkpoint + metrics`

**Prediction pipeline (runtime):**

`React file selection → image preview → multipart upload → FastAPI validation → Pillow RGB decode → exact training-time resize/normalization → trained checkpoint inference → JSON response → React result card`

The API must load the checkpoint once during application startup, not reload it for every request. The React application must not contain PyTorch or training logic. The original dataset must not be needed by the prediction server after a checkpoint has been created.

### 2.2 Project structure

Create the following layout. If the existing project has a different layout, adapt sensibly and document the mapping rather than deleting user files.

```text
FarmWatch/
├── README.md
├── steps.md
├── .gitignore
├── .env.example
├── requirements.txt
├── requirements-dev.txt
├── data/
│   ├── README.md
│   ├── raw/                    # downloaded/extracted original dataset; ignored by Git
│   └── processed/              # generated train/val/test data or split manifests; ignored by Git
├── docs/
│   ├── IMPLEMENTATION_STATUS.md
│   ├── DATASET_CARD.md
│   ├── MODEL_CARD.md
│   └── DECISIONS.md
├── scripts/
│   ├── inspect_dataset.py
│   ├── verify_dataset.py
│   └── prepare_dataset.py
├── training/
│   ├── train.py
│   ├── evaluate.py
│   └── model_utils.py
├── artifacts/                  # generated model and metrics; default ignored by Git
│   ├── models/
│   └── reports/
├── backend/
│   └── app/
│       ├── __init__.py
│       ├── main.py
│       ├── config.py
│       ├── schemas.py
│       ├── model_service.py
│       └── class_metadata.py
├── frontend/
│   ├── package.json
│   ├── index.html
│   ├── .env.example
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── App.css
│       ├── api.js
│       └── components/
│           ├── ImageUploader.jsx
│           ├── PredictionResult.jsx
│           └── ProbabilityList.jsx
└── tests/
    ├── test_dataset_utils.py
    ├── test_model_utils.py
    ├── test_api.py
    └── fixtures/                # tiny synthetic fixtures only for tests, never real model evidence
```

Keep the first implementation lean. The structure is a guide, not a reason to create lots of empty abstractions. If a module is unnecessary, combine it, but keep training, runtime inference, API routes, and UI components understandable and testable.

---

## 3. Phase 0 — Inspect the workspace before editing

1. Print the current working directory and list its contents.
2. Determine whether it is already a Git repository. If so, inspect `git status --short`, branch name, README, and dependency files. Never discard uncommitted changes.
3. Check installed Python and Node versions:
   - `python --version`
   - `node --version`
   - `npm --version`
4. Do not change global Python packages or Node installations. Use a project-local Python virtual environment.
5. If no project exists, create the structure above. If a project already exists, preserve code that is useful and fill the gaps.
6. Create/update `docs/IMPLEMENTATION_STATUS.md` with phases and checkboxes. Mark a phase complete only after its acceptance checks pass.
7. Keep a concise internal work log (for example, in `docs/IMPLEMENTATION_STATUS.md`) describing commands run and any issues. Do not put irrelevant terminal output into user-facing documentation.

**Acceptance checks:** workspace is understood; changes are non-destructive; status tracker exists.

---

## 4. Phase 1 — Python/Node setup and dependency files

### 4.1 Python environment

From the repository root:

```bash
python -m venv .venv
```

Activation commands:

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows Command Prompt**

```bat
.venv\Scripts\activate.bat
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

Create `requirements.txt` with the runtime and training packages actually used. Include at minimum FastAPI, Uvicorn, `python-multipart`, Pillow, NumPy, scikit-learn, Matplotlib, PyTorch, Torchvision, and any strictly necessary configuration library. Pin broad compatible minimums or use a tested lock strategy; don't guess a CUDA-specific PyTorch wheel inside the file.

Create `requirements-dev.txt` for `-r requirements.txt`, pytest, HTTPX, and optional lint/format tooling such as Ruff. Avoid installing duplicate variants of the same library.

Install PyTorch/Torchvision using the official selector: <https://pytorch.org/get-started/locally/>. The correct command depends on the operating system, Python version, and whether CUDA should be enabled. Then install the remaining project dependencies.

Verify:

```bash
python -c "import torch, torchvision; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('CUDA available:', torch.cuda.is_available())"
python -c "import fastapi, PIL, sklearn, numpy, matplotlib; print('Python dependencies OK')"
```

If CUDA is false, CPU training must still work. Never assume a GPU is installed. If a GPU exists but does not work, document the actual result and continue with a small CPU smoke test if feasible.

### 4.2 React environment

Create the React project using Vite if it does not already exist:

```bash
npm create vite@latest frontend -- --template react
cd frontend
npm install
cd ..
```

Run `npm run` inside `frontend/` and record which scripts actually exist. Use the generated Vite scripts rather than assuming a `lint` script was created.

### 4.3 Environment files and Git ignore

Create root `.env.example` and `frontend/.env.example` as appropriate. Never commit actual `.env` files. `VITE_` variables are public build-time values, so never place credentials in them.

Create `.gitignore` covering at least:

```gitignore
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.env
.env.*
!.env.example
frontend/node_modules/
frontend/dist/
data/raw/
data/processed/
artifacts/models/
artifacts/reports/
*.pth
*.pt
*.onnx
.DS_Store
Thumbs.db
```

If the team deliberately decides to version a small model artifact, revise the ignore rule intentionally and document model size/licensing. Do not accidentally track a model.

**Acceptance checks:** dependency files exist, Python imports succeed, Vite app installs, generated caches/data/secrets are ignored.

---

## 5. Phase 2 — Inspect and acquire the dataset

### 5.1 Inspect metadata before downloading

1. Open the dataset landing page and review the description, file list, version, licence, creators, and citation requirements.
2. Use Zenodo's API endpoint (`https://zenodo.org/api/records/19108517`) where network access permits. The inspection script should display record title/version, publication date, licence (if supplied), filenames, sizes, and download links without immediately downloading every file.
3. Identify the specific three-channel RGB archive or folder. Do not accidentally use the six-channel multispectral archive in a three-channel ResNet experiment.
4. Estimate disk use before extracting the archive. Check free disk space.
5. If network or provider access is unavailable, create `data/README.md` with the exact manual download procedure and continue implementing the preparation code against a documented expected input. Do not falsely mark the dataset as downloaded.
6. Avoid downloading the full 17.2 GB experimental drought dataset or full multi-gigabyte orthomosaic dataset for the MVP.

### 5.2 Dataset provenance

In `data/README.md` and `docs/DATASET_CARD.md`, record:

- official dataset title and record URL;
- access date and version;
- licence and citation instructions as shown by the source;
- selected archive filename, byte size, and SHA-256 checksum;
- image/channel format and expected labels;
- observed image count per class;
- original partitioning and metadata fields, if any;
- preprocessing/splitting policy;
- known limitations and any mismatch between expected and observed archive layout.

Compute SHA-256 for the downloaded archive (e.g. Python `hashlib` or a platform tool). If the record provides a checksum, verify it using the matching algorithm. A failed check must halt extraction with a clear error.

### 5.3 Dataset inspection script

Create `scripts/inspect_dataset.py` with a CLI. It should:

- accept a path to a directory, ZIP, or supported metadata file;
- list files and file extensions;
- count valid image files by folder where folder labels are available;
- safely inspect ZIP entries without extracting `../` paths or absolute paths;
- read image dimensions, colour mode, and decode failures with Pillow;
- display detected class folders and warn when an archive layout is not recognized;
- locate/read a CSV/JSON metadata file if present and print likely path/label/group columns (do not guess without showing the detected mapping);
- generate a machine-readable summary JSON in `artifacts/reports/dataset_inspection.json` (create the output directory if needed);
- avoid printing thousands of rows; show summaries and a few sample paths.

Example intended usage (adapt argument names to the script actually implemented):

```bash
python scripts/inspect_dataset.py --path data/raw
python scripts/inspect_dataset.py --path data/raw/rgb_subset.zip
```

### 5.4 Data layout support

The preparation utility must support, at minimum, a common folder-per-class layout:

```text
data/raw/<source-folder>/<class-name>/<image-file>
```

If the downloaded archive instead contains CSV labels, masks, nested class folders, or original `train`/`test` folders, adapt based on the real inspection results. Do not silently map a filename substring to a class unless the source documentation explicitly defines that convention.

### 5.5 Label mapping

Normalize spelling/case only in a controlled, documented way. Preserve the source label as `source_label` in the manifest. Map it to one of the supported canonical labels using an explicit mapping. For this dataset, the intended canonical labels are:

```text
healthy
low_stress
high_stress
rust
bare_soil
```

If any class name cannot be mapped unambiguously, report the unknown labels and stop preparation rather than placing them into `healthy` or dropping them silently.

**Acceptance checks:** the source is identified; archive/channel type is verified; label counts are visible; the licence and provenance are documented; no large files are accidentally added to Git.

---

## 6. Phase 3 — Dataset verification, deduplication, and splits

Create `scripts/verify_dataset.py` and `scripts/prepare_dataset.py`.

### 6.1 Validate images

For every candidate image:

1. Verify that Pillow can open and fully decode it.
2. Ignore non-image files such as `.txt`, `.json`, `.csv`, masks, and thumbnail files unless explicitly needed by the selected task.
3. Verify it has non-zero dimensions and a supported image mode.
4. Convert to RGB only at model loading unless a clean, documented preprocessing export makes that preferable.
5. Reject corrupt images with a report; do not quietly change their labels.
6. Count classes before and after exclusions.
7. Check for class imbalance. Report counts; don't invent samples to make classes balanced.

### 6.2 Detect duplicates and data leakage

This is important for field-image patches.

1. Calculate exact SHA-256 hashes of image files to detect byte-identical duplicates.
2. Where feasible, calculate perceptual hashes or another similarity diagnostic to flag differently encoded or resized near-duplicates. Do not blindly delete every similar-looking crop without manual/metadata review.
3. Inspect metadata for the original field, flight, trial, date, orthomosaic, plot, image, or source group ID.
4. Keep patches from the same source group in only one split where possible. Use a group-aware split when reliable source-group IDs exist.
5. If the public dataset ships official partitions, inspect and document them. Do not assume they are leakage-free; check duplicates across partitions. If the small subset explicitly contains samples drawn from upstream partitions and the source group is not available, document the limitation and run an exact/near-duplicate audit.
6. Never apply augmentation before splitting. Augmentation must happen only inside the training loader.

### 6.3 Split policy

Preferred split: group-aware 70% training / 15% validation / 15% test, approximately, stratified where the available library/data supports it. Preserve group separation even if exact percentages vary. Fix random seeds and save the split assignments into a manifest.

If group metadata does not exist, use a stratified image-level split, but:

- record that the evaluation may be optimistic because source-image grouping could not be guaranteed;
- perform duplicate checks before splitting;
- use a fixed random seed;
- never claim field-level generalization based on this alone.

Use a deterministic manifest such as `data/processed/splits.csv`, with columns like:

```text
relative_path,source_label,canonical_label,split,group_id,sha256
```

`group_id` may be empty only when no reliable group metadata exists. Write the manifest only after the mapping and split have been verified. Use relative paths and do not expose local absolute user paths in reports.

### 6.4 Expected processed layout

Either provide a `split.csv` manifest as the model's source of truth, or materialize folders in a consistent format:

```text
data/processed/
├── train/<class-name>/<image-file>
├── val/<class-name>/<image-file>
└── test/<class-name>/<image-file>
```

If files are copied, avoid overwriting identical basenames from different directories. Use a stable unique filename or preserve a safe nested relative path. Prefer a manifest-based loader if that is safer for the source archive.

### 6.5 Class count rules

- Print the total valid sample count and count per split/class.
- Warn if a class has too few examples to support a meaningful test estimate.
- Stop with a friendly explanation if a class is absent from training or validation.
- Do not assert accuracy is trustworthy if some classes have only a handful of examples.

**Acceptance checks:** split is reproducible; no exact duplicate crosses splits; group splitting is used where supported; counts and exclusions are recorded; split manifest exists.

---

## 7. Phase 4 — Implement and train the CNN with transfer learning

### 7.1 Model baseline

Implement a ResNet18 transfer-learning baseline with Torchvision ImageNet weights:

1. Create `models.resnet18(weights=ResNet18_Weights.DEFAULT)`.
2. Freeze the original feature extractor for the first training stage.
3. Replace the final fully connected layer with a dropout layer (optional) followed by a linear layer with output size `number_of_classes`.
4. Train the new head.
5. Unfreeze only the final residual block (`layer4`) and the classification head.
6. Fine-tune using a smaller learning rate than the head-only stage.
7. Use `CrossEntropyLoss`. If classes are imbalanced, select and document one strategy such as class-weighted loss or a weighted sampler; don't combine multiple balancing techniques without justification.
8. Save the best checkpoint based on **validation macro F1** (or a documented alternative chosen before test evaluation), not on test data.

Use pretrained weights only if they can be downloaded and license/source are known. If network is unavailable, fail with a clear instruction. Do not silently switch to random initialization and represent it as transfer learning. A random-initialized network may be used only for a smoke test.

### 7.2 Image preprocessing

Use a consistent input size such as 224 × 224 RGB. Keep training and inference preprocessing equivalent:

- Training: resize to the chosen image dimensions; use modest random horizontal flips, small rotations, and conservative brightness/contrast/saturation jitter if appropriate for these images; convert to tensor; normalize using ImageNet channel mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.
- Validation/test/inference: deterministic resize; tensor conversion; the same normalization; no random transforms.

Do not use aggressive transforms that change the meaning of water stress or alter crucial spectral/colour evidence. Do not apply ImageNet normalization twice. Put image size and normalization values in a shared/configured place and save them in checkpoint metadata.

### 7.3 Training script requirements

`training/train.py` must:

- take CLI arguments or configuration for dataset/manifest path, output directory, model architecture, epochs, batch size, learning rate, seed, number of workers, device, and early-stopping/patience settings if implemented;
- have sensible defaults for a small dataset and modest GPU memory (start with a small batch size such as 8, configurable);
- support CPU and CUDA selection safely;
- set Python, NumPy, and PyTorch random seeds;
- print device, class names, sample counts, class mapping, and important training settings;
- fail with actionable messages if data is missing, labels differ between splits, or a class has no samples;
- track train loss, validation loss, and validation macro F1 per epoch;
- save the best checkpoint, not just the last epoch;
- use checkpoint/model state saving that is compatible with the installed PyTorch version;
- write a JSON or CSV training history to `artifacts/reports/`;
- never read the test split for model selection;
- not hardcode a five-class output size—derive the output count from validated dataset metadata/class names.

Suggested defaults are a starting point, not guaranteed-optimal hyperparameters. Start with approximately 5 head-only epochs plus up to 8 fine-tuning epochs; lower batch size on limited GPU memory. Add early stopping if simple and correctly implemented. Do not spend hours seeking the best hyperparameters before the basic pipeline works.

### 7.4 Checkpoint contract

Save a checkpoint such as `artifacts/models/farmwatch_resnet18.pth` containing at least:

```text
model_state_dict
architecture
class_names                 # exact ordered list matching output neurons
class_metadata              # display name and category type, or a clear mapping key
image_size
normalization_mean
normalization_std
model_version
best_validation_macro_f1     # metric value and selection context
training_config
source_dataset_reference
```

Do not save a Python model object requiring fragile pickle imports when a state dictionary and explicit architecture are sufficient. Use `map_location` when loading. Test checkpoint loading in a fresh process.

### 7.5 Training command and output

Document the exact command used after the dataset loader format is implemented. Example interface:

```bash
python -m training.train --data-dir data/processed --epochs 13 --batch-size 8 --seed 42
```

The code may choose different argument names, but update the README with the actual supported command.

Expected artifacts:

```text
artifacts/models/farmwatch_resnet18.pth
artifacts/reports/training_history.csv
artifacts/reports/training_curves.png
artifacts/reports/dataset_inspection.json
```

If Matplotlib is used to plot curves, generate at least train/validation loss and validation macro F1. Label axes and include the saved image size/device context where useful.

**Acceptance checks:** real dataset used; training starts; validation runs; best checkpoint saved; metadata is complete; logs/history exist; no test data is used to choose the model.

---

## 8. Phase 5 — Evaluate the trained checkpoint properly

Create `training/evaluate.py`. It must load the saved best checkpoint and evaluate the test split once the training configuration/model selection is complete.

Generate:

1. Test loss.
2. Overall accuracy (secondary metric, not the only metric).
3. Macro precision, macro recall, and **macro F1**.
4. Per-class precision, recall, F1, and support.
5. Confusion matrix with readable labels.
6. A machine-readable JSON results file and a CSV classification report.
7. A confusion-matrix PNG.
8. A note explaining the test split methodology and possible limitations.

Use all class labels in the correct saved class order so missing prediction classes do not shift the matrix axes. Guard divisions/zero counts safely. Report the number of test examples beside metrics.

Never write fixed/claimed percentages into documentation before the actual run. The README must use a placeholder such as “Run evaluation to produce metrics” until a legitimate trained model was evaluated. If the subset is too small, show counts and warn that metrics have high uncertainty. Do not evaluate repeatedly on test and tune against the outcome.

Optional, only after the baseline passes: compare against a simple pretrained EfficientNet-B0/MobileNetV3 baseline using the same splits. Compare macro F1, inference speed, and model size using identical conditions. Do not add a second architecture if it makes the initial training/API pipeline unstable.

**Acceptance checks:** evaluation produces metrics from the stored checkpoint; all five classes are accounted for; reports match the held-out test count; no results are fabricated.

---

## 9. Phase 6 — Build an inference service that can be tested independently

Create `backend/app/model_service.py` to own model loading and image inference. Keep model construction/loading separate from FastAPI route definitions so tests can substitute a fake predictor without requiring a trained checkpoint.

### 9.1 Model service responsibilities

- Load the configured checkpoint once at startup.
- Validate required checkpoint keys and that class names are a non-empty ordered list.
- Construct the correct architecture from the checkpoint metadata. For the MVP, explicitly support `resnet18`; fail clearly for unsupported architectures.
- Load the state dict strictly wherever possible. Do not quietly ignore incompatible keys.
- Set `model.eval()` and place the model on an available CUDA device or CPU.
- Use `torch.inference_mode()` for inference.
- Convert the decoded input to RGB and apply exactly the deterministic preprocessing used at evaluation time.
- Return the top predicted class and softmax scores in the exact saved class order.
- Handle inference failures without returning raw stack traces to public API clients.

### 9.2 Runtime settings

Create a config module that reads environment variables such as:

- `FARMWATCH_MODEL_PATH` — default path to the exported model artifact;
- `FRONTEND_ORIGINS` — comma-separated allowed origins, default local Vite origin during development;
- `MAX_UPLOAD_MB` — default 10;
- `MAX_IMAGE_PIXELS` — choose a reasonable bound to reduce decompression-bomb risk;
- an optional confidence-warning threshold, clearly documented as a provisional UX threshold unless calibrated.

Validate configuration on startup and report actionable errors in the server log. Avoid hard-coded production domains in the source code.

### 9.3 Image upload validation

The prediction endpoint must not trust the file extension or client MIME type by itself.

1. Read the request body with a bounded strategy. Do not read unlimited bytes into memory and only then check the size.
2. Reject empty files, files over the configured size limit, invalid/corrupt images, unsupported formats, and images exceeding a configured pixel/dimension limit.
3. Use Pillow to verify/decode the image and convert it to RGB.
4. Return a safe, clear 4xx error for invalid input; do not expose stack traces or server file paths.
5. Do not save uploads permanently for the MVP. Discard image bytes after inference.
6. Where practical, handle Pillow decompression-bomb warnings/errors explicitly.

The frontend should accept JPEG/PNG for the MVP and explain the size limit. The backend remains authoritative for validation.

### 9.4 API routes

Implement and document:

- `GET /` — short service name/version and API documentation pointer.
- `GET /health` — liveness and model-loaded status. Do not report healthy model status if weights failed to load.
- `GET /api/classes` — canonical labels, human-readable labels, category types, and short descriptions.
- `POST /api/predict` — multipart image field named `file`; returns prediction JSON.

A successful prediction response should have a stable shape similar to:

```json
{
  "predicted_class": "low_stress",
  "display_name": "Low water-stress class",
  "category_type": "water_stress",
  "confidence": 0.73,
  "confidence_warning": true,
  "probabilities": {
    "healthy": 0.08,
    "low_stress": 0.73,
    "high_stress": 0.11,
    "rust": 0.05,
    "bare_soil": 0.03
  },
  "interpretation": "The image resembles the low-stress class in the training dataset.",
  "model_version": "..."
}
```

This is an example contract only. Actual numbers must come from the model; do not hardcode sample probabilities into the route. Use a Pydantic response model or equivalent schema validation. Return all class probabilities in the checkpoint's fixed order. Use consistent names across backend/frontend.

### 9.5 Confidence and out-of-distribution caution

- `softmax` gives relative model scores; these are not guaranteed calibrated probabilities.
- A simple warning threshold can be used for initial UX, but label it as a provisional warning, not proven uncertainty detection.
- If calibration is added, fit/validate it only on training/validation data (never on test outcomes) and document the method.
- A five-class softmax must always select one class even for a dog, person, blurred photo, or unrelated plant. Do not claim it reliably rejects those images without a separate validated rejection strategy. For the MVP, display an explicit notice to upload a clear image of maize and explain that out-of-scope inputs may get misleading predictions.
- Optionally add basic image-quality checks (e.g. empty/tiny image) but do not pretend a simple heuristic proves an image is in-domain.

### 9.6 CORS and API security basics

- Configure CORS from environment variables; allow only expected origins.
- Do not use `allow_origins=["*"]` with credentialed requests.
- Do not log uploaded image bytes, base64 image data, secrets, or unnecessary personal information.
- Limit upload size and decode dimensions; consider a modest request-rate limit before a public release.
- Do not add account/login/database functionality unless requested; the MVP needs none.

### 9.7 Running the backend

From the repository root, document and verify a command similar to:

```bash
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

The actual import path must correspond to the package structure. Test `http://127.0.0.1:8000/health` and `http://127.0.0.1:8000/docs`. If the model checkpoint does not exist, the server should fail with an understandable startup error and tell the developer which training command to run. Never fall back to a random model in normal development/production.

**Acceptance checks:** model loads one time; API schemas match response JSON; file validation works; CORS works; invalid uploads receive safe errors; inference can be unit tested without a real model checkpoint.

---

## 10. Phase 7 — Build the React frontend

### 10.1 Visual direction

Create a polished, minimal agricultural AI interface, not a generic unstyled form. Use a natural, restrained palette: deep forest green, muted sage, warm off-white, and neutral gray. Keep contrast readable. Use CSS variables, consistent spacing, restrained borders, and clear typography. Avoid excessive gradients, glassmorphism, oversized animations, emoji-heavy presentation, and decorative elements that compete with the prediction.

The interface should work on desktop and mobile, and the upload/prediction flow should be obvious without onboarding.

### 10.2 Required page content

Implement a single main analysis page containing:

1. Header: FarmWatch wordmark, compact “Crop condition analysis” description, and a visible “AI estimate, not a diagnosis” notice.
2. Image upload panel with drag-and-drop and a normal file chooser; accept only JPEG/PNG; enforce the client-side 10 MB limit; explain accepted formats.
3. Image preview with a remove/replace action. Revoke object URLs when the selected image changes or the component unmounts to avoid browser memory leaks.
4. Primary action button such as **Analyze image**. Disable it when no file is selected or a request is running.
5. Loading state that communicates progress without pretending to know a percentage.
6. Clear error states for API offline, invalid file, oversized image, invalid image bytes, HTTP error, and general network failures. Provide a retry action where appropriate.
7. Results panel with predicted label, type/category badge, model confidence, cautious interpretation, class probability bars, and a “Analyze another image”/reset action.
8. Low-confidence warning when backend policy signals it. Explain that a model score is not a diagnosis.
9. A small methodology note: trained on the documented selected maize dataset and not validated for all crops or all smartphone/field conditions.
10. Accessible semantics: real labels for the input; buttons with accessible names; visible focus state; meaningful image alt text; color not the only status cue; sufficient contrast; status/errors announced accessibly when appropriate.

### 10.3 Component responsibilities

- `App.jsx`: page layout and top-level state.
- `ImageUploader.jsx`: file selection/drop, format/size checks, preview lifecycle, remove file.
- `PredictionResult.jsx`: predicted label, confidence/warning, interpretation, disclaimer.
- `ProbabilityList.jsx`: sorted class probability bars with numeric percentages and accessible labels.
- `api.js`: shared fetch function, API base URL, typed-by-convention request/response handling, and parsing useful error messages.
- `App.css`: responsive style system, design tokens, states, mobile layout.

Keep state minimal: selected file, preview URL (can be derived in the uploader component), prediction response, loading flag, and error message. Clear stale prediction results when a new file is selected. Prevent duplicate simultaneous submissions. Do not persist image uploads to local storage or any database.

### 10.4 API integration

Create `frontend/.env.example` containing:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Use `import.meta.env.VITE_API_BASE_URL` with a safe documented development default. When uploading with `fetch`, send `FormData` with a `file` field. Do **not** manually set the `Content-Type` header to `multipart/form-data`; the browser must set the multipart boundary automatically.

Handle non-2xx status codes. FastAPI error responses may use a `detail` string or validation object; display a readable message rather than `[object Object]` or a raw stack trace.

Do not hardcode the probabilities from the example API response. Render only values received from the backend. Normalize/format class labels through the class metadata mapping, not ad-hoc string replacements when a mapping exists.

### 10.5 Frontend scripts and checks

Ensure `frontend/package.json` has usable scripts (at minimum `dev`, `build`, and any lint/test scripts that are actually configured). Run:

```bash
cd frontend
npm install
npm run build
```

Add a lint script/tool if reasonable and fix all newly introduced lint errors. Do not state that lint passed if no lint script exists or was not run.

**Acceptance checks:** frontend build succeeds; upload works; preview and reset work; API request uses correct form field/base URL; loading/error/result states render; no fake model outputs are shown.

---

## 11. Phase 8 — End-to-end integration

Run the backend and frontend in separate terminals.

**Terminal A — backend** (activate `.venv` first):

```bash
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

**Terminal B — frontend:**

```bash
cd frontend
npm run dev
```

Open the local URL printed by Vite (commonly `http://localhost:5173`). Confirm that `VITE_API_BASE_URL` and backend CORS origins agree. If the frontend environment file was created/changed after Vite started, restart the Vite process.

Test at least:

1. A legitimate image from each available class in the dataset's held-out test set.
2. An image that is not an allowed format.
3. An oversized file.
4. A corrupt file renamed `.jpg`.
5. No file selected.
6. Backend stopped while frontend remains open.
7. A second image chosen after one prediction (old result should clear).
8. Repeated prediction requests (server should remain responsive).
9. Narrow/mobile viewport.
10. Direct API request via `/docs` or a small curl command.

**Important:** a correct end-to-end response proves that the software pipeline works; it does not prove that the model is accurate. Use test-set evaluation and field-image checks for model quality.

---

## 12. Phase 9 — Automated tests and code quality

Create pytest tests for critical paths. Separate pure utilities and the model service from the API so most tests do not need the large dataset or a GPU.

### 12.1 Dataset tests

- Recognized classes map to canonical labels correctly.
- Unknown labels raise a clear error.
- Corrupt/non-image files are reported and excluded according to policy.
- Exact duplicates are detected.
- Split assignments are deterministic for the same seed.
- When a reliable group ID exists, no group appears in multiple splits.
- Manifest paths are relative/safe and files exist.

### 12.2 Training/model tests

- Model factory returns the expected number of output classes.
- Preprocessing returns a 3-channel tensor of the expected dimensions.
- Checkpoint metadata preserves exact output label order.
- Checkpoint save/load round trip succeeds with a tiny test configuration.
- A fake/tiny model can run one forward pass and return finite logits.
- The tests must not silently replace the real checkpoint required by the live API.

### 12.3 API tests

- Health route returns the correct service/model state.
- Classes route returns valid class metadata.
- Valid image upload returns the agreed response schema.
- Invalid image bytes return 4xx, not an unhandled 500.
- Empty/oversized files are rejected.
- Wrong methods or missing `file` produce expected HTTP errors.
- The prediction probabilities have valid values and sum approximately to 1 (allow small floating-point tolerance).
- Use an injected fake predictor for test isolation; never claim the fake model's prediction is model evaluation.

### 12.4 Frontend checks

- `npm run build` succeeds.
- Any configured lint/test command succeeds.
- Upload errors, loading state, and result rendering are tested if practical.

### 12.5 Suggested commands

Document and run commands that match the actual implementation:

```bash
python -m pytest -q
```

```bash
cd frontend
npm run build
```

If a quality tool is configured, also run its real command (for example Ruff). Do not suppress errors by broadly disabling rules. Fix root causes where reasonable.

**Acceptance checks:** critical logic is tested without downloading/training the full dataset; build/tests pass or documented external blockers are specific.

---

## 13. Phase 10 — Documentation and reproducibility

### `README.md` must include

1. What FarmWatch does and what it does **not** do.
2. Screenshots only if actually produced; never fake UI screenshots.
3. Tech stack and architecture overview.
4. Dataset source, licence, classes, citation, limitations, and the manual download steps.
5. Prerequisites and verified Python/Node versions.
6. Virtual environment and dependency installation commands for Windows PowerShell and macOS/Linux.
7. Commands to inspect, verify, prepare, train, and evaluate the dataset.
8. Where the trained checkpoint and metrics are written.
9. Commands to start the backend and frontend in separate terminals.
10. API endpoint descriptions and an example response schema clearly marked as illustrative.
11. Test commands.
12. Deployment notes and required environment variables.
13. Known issues and appropriate limitations, especially small dataset size and image-domain mismatch.
14. How to cite the dataset and pretrained weights according to their source terms.

### `docs/MODEL_CARD.md` must include

- model architecture and pretrained weight source;
- input format and image preprocessing;
- ordered class list and meanings;
- training/validation/test methodology;
- measured metrics with test support count only after actual evaluation;
- training hardware/device where known;
- intended and non-intended uses;
- known failure modes and risks of domain shift;
- date/version of the model artifact.

### `docs/DATASET_CARD.md` must include

- source, licence, citation, acquisition/version date;
- class distribution and image counts;
- image resolution/channel details;
- source group information and data leakage controls;
- train/validation/test split details;
- known issues such as small sample size, background/field bias, dataset taxonomy limitations, or missing group IDs.

### Reproducibility

Set random seeds and save the resolved config. Write the exact command used to produce the checkpoint. If results are not reproducible exactly because of CUDA/nondeterministic kernels, document this rather than claiming perfect determinism. Do not commit a huge archive or sensitive local path.

---

## 14. Phase 11 — Deployment readiness

Do not begin deployment before local inference works.

### 14.1 Frontend

- Build the Vite application using `npm run build`.
- Set `VITE_API_BASE_URL` to the public HTTPS API origin in the deployment platform's build environment.
- Confirm this variable is public and contains no secret.
- Deploy the generated build using Vercel, Netlify, or a comparable static host.

### 14.2 Backend

- Pick a Python-compatible host with enough RAM/disk for PyTorch and the checkpoint. Confirm current provider limits instead of assuming a free tier can run PyTorch reliably.
- Keep `FARMWATCH_MODEL_PATH` correct relative to the deployed filesystem or place the model at a documented mounted/downloaded path.
- Set `FRONTEND_ORIGINS` to the exact deployed frontend origin(s), comma-separated if multiple are intentionally used.
- Set upload/pixel limits and any warning threshold deliberately.
- Use HTTPS. Do not expose debug mode or detailed tracebacks publicly.
- Confirm that the model artifact is actually present in the deployment runtime; otherwise health must not claim that the model is loaded.
- If the artifact is too big for the repository, use the platform's supported artifact/object storage flow and verify the download/checksum at startup. Do not put credentials in source code.
- Add health checks and inspect actual server startup logs after deployment.

### 14.3 Production limitations

The starter app is intentionally stateless; uploads are not saved. If persistent prediction history is later requested, plan privacy, retention, consent, storage costs, and data deletion explicitly before adding a database.

**Acceptance checks:** deployment build passes, the hosted API loads the actual trained model, CORS works from the hosted UI, and one prediction succeeds over HTTPS.

---

## 15. Phase 12 — Optional enhancements (only after the MVP passes)

Implement in this order and only if the core app and tests are stable:

1. **Grad-CAM visualization:** show an approximate heatmap for model debugging; explain that it is an attribution visualization, not a verified map of plant pathology.
2. **Model comparison:** train EfficientNet-B0 or MobileNetV3 on identical splits and report macro F1, inference time, and artifact size.
3. **Three-class stress-only experiment:** train a separately named model using only `healthy`, `low_stress`, and `high_stress`, while retaining a clear policy for rust/bare-soil/unknown photos. Do not silently drop those labels from the five-class checkpoint or confuse its output taxonomy.
4. **Field segmentation:** use the full dataset's pixel-level masks only after inspecting the annotation format and licence; implement a segmentation model as a separate task from image-level classification.
5. **Prediction history:** only if requested; requires explicit decisions about storage and retention.

Do not start with all enhancements. Deliver and verify the small functional MVP first.

---

## 16. Troubleshooting guide

### PyTorch or Torchvision installation mismatch

- Use the official PyTorch installation selector for the current system.
- Confirm imports and printed versions.
- Do not randomly downgrade packages. If an operator mismatch occurs, install compatible Torch/Torchvision builds as a pair.

### CUDA unavailable or out-of-memory

- CPU mode must remain supported.
- Reduce batch size (e.g. 8 → 4 → 2), reduce worker count, and ensure only the model/data batch is on the GPU.
- Do not reduce validation sample counts or alter labels to improve results.

### Dataset folders do not match the expected structure

- Run the inspection script first.
- Adapt the data loader/manifest mapping to observed filenames and metadata.
- Do not guess classes from image appearance.
- Update `data/README.md` with the actual mapping.

### Dataset downloads fail

- Try the record landing page and its listed file links/API metadata.
- Do not repeatedly download the entire record if only an RGB subset is needed.
- If still unavailable, document the file the user must download manually and make sure the rest of the software can be tested using fixtures.

### Training has suspiciously high accuracy

- Check exact and near duplicates across splits.
- Check whether adjacent image patches or the same field/flight were split across train and test.
- Check whether labels leaked through filenames, folder paths, metadata, or embedded overlays.
- Report the split limitation honestly; do not publish the score as reliable field accuracy.

### Predictions are consistently one class

- Print class counts and target mapping.
- Check label indices align with output neuron order in the checkpoint.
- Inspect class imbalance and loss configuration.
- Verify RGB conversion and normalization are the same at training and inference.
- Inspect a batch and logits in a controlled test; do not patch the prediction response with a hardcoded class.

### Frontend gets a CORS or network error

- Confirm FastAPI is running and the correct port is used.
- Confirm `VITE_API_BASE_URL` has no unintended trailing path and points to the API origin.
- Confirm the browser origin is included in `FRONTEND_ORIGINS`.
- Restart Vite after editing env variables.
- Never “fix” production CORS by allowing every origin with credentials.

### API says model not found

- Check that training actually produced the checkpoint.
- Check `FARMWATCH_MODEL_PATH` and working-directory assumptions.
- Keep model paths derived from module/config locations rather than relying on whichever directory started Uvicorn.

### The model returns confident predictions for unrelated photos

- This is expected for a closed-set classifier unless a validated rejection strategy exists.
- Improve the UI disclaimer and data domain; consider a separate in-domain/quality detector only after collecting suitable validation data.
- Do not describe a fixed softmax threshold as a guaranteed OOD detector.

---

## 17. Final acceptance checklist — do not claim completion until checked

### Repository and documentation

- [ ] Project structure exists and existing user work was preserved.
- [ ] `.gitignore` excludes environment, data, generated artifacts, and secrets.
- [ ] README setup instructions match the actual commands.
- [ ] Dataset provenance/licence/class mapping are documented.
- [ ] Model card and implementation status are present.

### Data

- [ ] The actual dataset archive/layout has been inspected.
- [ ] Valid image and class counts are recorded.
- [ ] Unknown/corrupt images are reported.
- [ ] Exact duplicate files are audited; near-duplicate checks are performed where feasible.
- [ ] Splits are deterministic and use group separation when source metadata supports it.
- [ ] Split manifest and methodology report exist.

### Model

- [ ] ResNet18 pretrained transfer-learning baseline is implemented.
- [ ] Training works on CPU; CUDA is used only when genuinely available.
- [ ] Best checkpoint is chosen using validation, not test, metrics.
- [ ] Checkpoint contains correct ordered labels and preprocessing metadata.
- [ ] Evaluation report includes test count, macro F1, per-class scores, and confusion matrix.
- [ ] No performance claims have been invented or taken from training accuracy as final proof.

### Backend

- [ ] Checkpoint loads once at startup.
- [ ] `/health`, `/api/classes`, and `/api/predict` work.
- [ ] API errors are safe and understandable.
- [ ] Image bytes, image dimensions, MIME/decoding, and upload size are validated.
- [ ] Inference preprocessing is identical to evaluation preprocessing.
- [ ] CORS is configurable and not overly permissive.

### Frontend

- [ ] Vite production build passes.
- [ ] User can choose/drop a file, see a preview, analyze it, and reset.
- [ ] Loading, error, empty, success, and low-confidence states work.
- [ ] Probabilities come from the API, not hardcoded demo numbers.
- [ ] Responsive layout and accessible labels/focus states are implemented.
- [ ] Disclaimer and dataset limitations are visible.

### Testing and integration

- [ ] Python tests pass or remaining blockers are specified precisely.
- [ ] Frontend build/lint/tests that exist were run and results recorded.
- [ ] At least one real held-out dataset image completed the full React → API → trained-model pipeline, if the dataset/model download was possible.
- [ ] Invalid file and backend-offline behavior were checked.
- [ ] Actual tested commands and remaining work are summarized for the user.

---

## 18. Expected final report from Antigravity

When implementation work is finished, report concisely:

1. What was implemented and the important files created/modified.
2. Dataset source/version used (or exact manual download step if blocked).
3. Exact command to prepare the dataset, train, evaluate, start backend, and start frontend.
4. Whether a **real trained checkpoint** was created. Include its path/size only if verified.
5. Actual test results: commands run and pass/fail counts. Clearly separate code smoke tests from evaluation on the real dataset.
6. Actual model metrics only if the held-out test evaluation was run; include sample count and split limitations.
7. Remaining blockers or manual actions, with the exact filenames/commands needed.

Do not claim the app is “fully working” if the model was not trained or the React → FastAPI → model path was not tested. Deliver the best verified state, explain exactly what is missing, and do not conceal failures.
