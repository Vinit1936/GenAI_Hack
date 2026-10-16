# Architectural and Engineering Decisions

## 1. Python Environment Selection
- **Decision:** Use Python 3.11 (`py -3.11`) for the project virtual environment.
- **Rationale:** While Python 3.13 is installed as system default, Python 3.11 provides complete, mature binary wheel compatibility with PyTorch, Torchvision, and scientific packages (`scikit-learn`, `scipy`) on Windows without compilation issues.

## 2. Model Baseline Architecture
- **Decision:** ResNet-18 pretrained on ImageNet from `torchvision.models`.
- **Rationale:** Efficient, fast transfer-learning convergence on small patch datasets, low GPU/CPU memory footprint, and reliable feature extraction for visible agricultural stress patterns.

## 3. Dataset & Label Taxonomy
- **Decision:** Use the 5 canonical classes from the Zenodo 19108517 RGB subset: `healthy`, `low_stress`, `high_stress`, `rust`, `bare_soil`.
- **Rationale:** Preserves ground truth fidelity without artificially forcing background or rust classes into water-stress severity bins.

## 4. Split Strategy
- **Decision:** Group-aware stratified split (~70% train / 15% val / 15% test) where flight/field metadata allows, backed by exact SHA-256 duplicate auditing.
- **Rationale:** Prevents optimistic accuracy inflation and data leakage caused by spatial proximity of patch samples.

## 5. Inference Service & Model Lifetime
- **Decision:** Load checkpoint once at FastAPI startup (`lifespan` handler) into memory; keep inference stateless.
- **Rationale:** Minimizes latency, prevents memory leaks from repeated model initialization, and preserves stateless upload safety.
