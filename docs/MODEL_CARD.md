# Model Card: FarmWatch ResNet-18 Baseline

## 1. Model Details
- **Architecture:** ResNet-18 (`torchvision.models.resnet18`)
- **Weights:** ImageNet pre-trained weights (`ResNet18_Weights.DEFAULT`)
- **Fine-Tuning:** Two-stage transfer learning:
  1. Head-only training (frozen backbone, AdamW, lr=1e-3, 4 epochs)
  2. Backbone fine-tuning (`layer4` + classification head unfrozen, AdamW, lr=1e-4, 4 epochs)
- **Selection Metric:** Best validation Macro F1 score (achieved 0.9544 at Epoch 8)
- **Artifact File:** `artifacts/models/farmwatch_resnet18.pth` (44.8 MB)
- **Date & Version:** 2026-10-16 / Version 1.0.0

## 2. Intended Use
- Classification of maize canopy condition from RGB imagery into 5 classes: `healthy`, `low_stress`, `high_stress`, `rust`, `bare_soil`.
- Visual assistive estimate tool for agronomists, field researchers, and growers.
- Not a definitive or certified diagnosis for pesticide or chemical treatment prescription.

## 3. Input Specification
- **Format:** RGB image (JPEG, PNG, TIFF)
- **Dimensions:** Resized to 224 x 224 pixels
- **Normalization:** ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`

## 4. Output Specification
- **Predicted Class**: Top canonical label (`bare_soil`, `healthy`, `high_stress`, `low_stress`, `rust`)
- **Display Label**: Human-readable name and category classification
- **Probabilities**: Softmax probability distribution across all 5 classes in checkpoint order
- **Confidence Warning**: Flagged if top confidence < 0.60

## 5. Measured Performance on Held-Out Test Set (114 Samples)
Evaluated strictly once on the held-out test split (zero leakage from exact duplicate clusters):
- **Overall Accuracy:** 96.49%
- **Macro Precision:** 0.9673
- **Macro Recall:** 0.9648
- **Macro F1 Score:** 0.9655
- **Test Loss:** 0.0819

### Per-Class Test Breakdown
| Class | Category Type | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `bare_soil` | background | 1.0000 | 1.0000 | 1.0000 | 23 |
| `healthy` | plant_condition | 1.0000 | 0.9565 | 0.9778 | 23 |
| `high_stress` | water_stress | 0.8800 | 0.9565 | 0.9167 | 23 |
| `low_stress` | water_stress | 0.9565 | 0.9565 | 0.9565 | 23 |
| `rust` | disease | 1.0000 | 0.9545 | 0.9767 | 22 |

## 6. Out-of-Distribution & Risk Disclaimers
- Closed-set classifier: Images of non-maize plants, weeds, or irrelevant objects will still be classified into one of the 5 classes.
- Aerial UAV perspective: Trained on orthorectified UAV patch crops; smartphone handheld field photos may exhibit optical and angle domain shifts.
- Softmax outputs represent relative resemblance to dataset clusters, not calibrated real-world probabilities.
