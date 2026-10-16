# Dataset Card: Zenodo Record 19108517 (Maize Water Stress & Condition RGB Subset)

## 1. Summary
- **Title:** Field Maize Condition / Water Stress RGB Image Patches
- **Record Identifier:** Zenodo Record 19108517
- **Landing Page:** https://zenodo.org/records/19108517
- **API URL:** https://zenodo.org/api/records/19108517
- **Domain:** Aerial / UAV agricultural imaging of maize crops under varied water-stress, disease, and ground conditions.

## 2. Taxonomy & Classes
The dataset RGB subset comprises up to 100 image patches per class across 5 canonical classes:
1. `healthy` (display: "Healthy-looking maize"): Maize foliage showing no visible severe stress or chlorosis.
2. `low_stress` (display: "Low water-stress class"): Early or mild drought/water stress symptoms.
3. `high_stress` (display: "High water-stress class"): Pronounced drought symptoms, severe wilting/curling.
4. `rust` (display: "Rust-disease class"): Fungal rust infection on maize foliage.
5. `bare_soil` (display: "Bare-soil/background class"): Soil background, non-crop patch.

## 3. Image Characteristics
- **Channels:** 3-channel RGB
- **Input Dimensions:** Patches standardized to 224 x 224 pixels during training.
- **Normalization:** ImageNet statistics (Mean: `[0.485, 0.456, 0.406]`, Std: `[0.229, 0.224, 0.225]`).

## 4. Split and Leakage Control
- **Splits:** 70% Train / 15% Validation / 15% Test.
- **Auditing:** Exact SHA-256 duplicate auditing.
- **Group Separation:** Patches from identical flight/field groups are assigned to a single partition where source metadata permits.

## 5. Limitations
- Limited sample size (~100 patches per class).
- Specific to maize UAV domain; not validated across arbitrary crops or variable smartphone camera optics without domain adaptation.
- Softmax outputs represent category resemblance within this dataset, not calibrated agricultural diagnoses.
