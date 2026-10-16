# Model Card: FarmWatch ResNet-18 Baseline

## 1. Model Details
- **Architecture:** ResNet-18 (`torchvision.models.resnet18`)
- **Weights:** ImageNet pre-trained weights (`ResNet18_Weights.DEFAULT`)
- **Fine-Tuning:** Two-stage transfer learning:
  1. Head-only training (frozen backbone) with `CrossEntropyLoss`.
  2. Backbone fine-tuning (`layer4` unfrozen) with lower learning rate.
- **Checkpoint Contract:** PyTorch `.pth` checkpoint containing state dictionary, class names, preprocessing parameters, model version, and validation metrics.

## 2. Intended Use
- Classification of maize canopy condition from RGB imagery into 5 classes: `healthy`, `low_stress`, `high_stress`, `rust`, `bare_soil`.
- AI assistive estimation tool; not a definitive agronomic diagnosis.

## 3. Input Specification
- **Format:** RGB image (JPEG or PNG)
- **Dimensions:** Resized to 224 x 224 pixels
- **Normalization:** ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`

## 4. Output Specification
- **Top predicted class**: Canonical label and display name.
- **Probabilities**: Softmax probability distribution across all 5 classes.
- **Confidence warning**: Flagged if top confidence falls below configured threshold (default: 0.60).

## 5. Performance Metrics
- *Metrics will be populated following model training and evaluation on the held-out test split.*
- Target selection metric: Validation Macro F1.

## 6. Out-of-Distribution & Risk Disclaimers
- Closed-set classifier: Images outside the 5 training categories (unrelated objects, weeds, other crops) will still map to one of the 5 classes.
- Should never be used as the sole basis for applying chemical treatments or irrigation decisions.
