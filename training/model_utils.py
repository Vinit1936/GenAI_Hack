"""FarmWatch Model Utilities and Preprocessing Pipelines.

Provides architecture setup (ResNet-18 transfer learning),
ImageNet transforms, PyTorch Dataset, and robust checkpoint contracts.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torchvision.transforms as transforms

# Standard ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_IMAGE_SIZE = 224

# Class metadata definition with precise agronomic and disease taxonomy
CLASS_METADATA = {
    "healthy": {
        "display_name": "Healthy Maize Foliage",
        "category_type": "plant_condition",
        "disease_name": "None (Healthy Crop)",
        "scientific_name": "N/A",
        "pathogen_type": "None",
        "host_crop": "Maize / Corn (Zea mays)",
        "symptoms": "Uniform green canopy with intact chlorophyll distribution, no wilting, and zero lesion development.",
        "description": "Image resembles healthy maize foliage without visible severe stress, chlorosis, or disease pustules.",
    },
    "low_stress": {
        "display_name": "Mild Water Stress (Early Drought)",
        "category_type": "water_stress",
        "disease_name": "Abiotic Moisture Stress (Mild)",
        "scientific_name": "N/A (Physiological Drought)",
        "pathogen_type": "Environmental Deficit",
        "host_crop": "Maize / Corn (Zea mays)",
        "symptoms": "Subtle grayish-green leaf tint, upward rolling of leaf margins during peak daylight hours, and slight turgor loss.",
        "description": "Image resembles early or mild moisture stress in maize crops.",
    },
    "high_stress": {
        "display_name": "Severe Water Stress (Acute Drought)",
        "category_type": "water_stress",
        "disease_name": "Abiotic Moisture Stress (Severe)",
        "scientific_name": "N/A (Physiological Drought)",
        "pathogen_type": "Environmental Deficit",
        "host_crop": "Maize / Corn (Zea mays)",
        "symptoms": "Pronounced inward leaf rolling (pencil-rolling), tip burning/senescence, and severe canopy wilting.",
        "description": "Image resembles pronounced drought stress with severe moisture deficit and wilting.",
    },
    "rust": {
        "display_name": "Common Corn Rust (Puccinia sorghi)",
        "category_type": "disease",
        "disease_name": "Common Corn Rust",
        "scientific_name": "Puccinia sorghi",
        "pathogen_type": "Fungal Pathogen (Basidiomycota)",
        "host_crop": "Maize / Corn (Zea mays)",
        "symptoms": "Small, circular to elongate golden-brown to cinnamon pustules (uredinia) scattered across both upper and lower leaf surfaces that rupture the epidermis and release powdery spores.",
        "description": "Foliar fungal infection on maize caused by Puccinia sorghi; distinct pathology characterized by powdery rust pustules.",
    },
    "bare_soil": {
        "display_name": "Bare Soil / Ground Background",
        "category_type": "background",
        "disease_name": "Non-Crop Background",
        "scientific_name": "N/A",
        "pathogen_type": "None (Non-foliar)",
        "host_crop": "None (Soil)",
        "symptoms": "Exposed agricultural earth, tilled soil, or mulch without active crop vegetation.",
        "description": "Image resembles exposed field ground or soil background; non-crop patch.",
    },
}


def get_transforms(image_size: int = DEFAULT_IMAGE_SIZE) -> Tuple[transforms.Compose, transforms.Compose]:
    """Get training (with moderate augmentation) and validation/inference transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, eval_transform


class FarmWatchDataset(Dataset):
    """PyTorch Dataset loading images using relative paths and canonical labels."""

    def __init__(
        self,
        base_dir: Path,
        records: List[Dict[str, str]],
        class_to_idx: Dict[str, int],
        transform: Optional[transforms.Compose] = None,
    ):
        self.base_dir = Path(base_dir)
        self.records = records
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        rec = self.records[idx]
        img_path = self.base_dir / rec["relative_path"]
        with Image.open(img_path) as img:
            img = img.convert("RGB")

        if self.transform is not None:
            tensor = self.transform(img)
        else:
            tensor = transforms.ToTensor()(img)

        target = self.class_to_idx[rec["canonical_label"]]
        return tensor, target


def load_split_records(
    manifest_path: Path, split_name: str
) -> Tuple[List[Dict[str, str]], List[str], Dict[str, int]]:
    """Load records for a given split from data/processed/splits.csv."""
    records = []
    classes_set = set()

    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            classes_set.add(row["canonical_label"])
            if row["split"] == split_name:
                records.append(row)

    class_names = sorted(classes_set)
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    return records, class_names, class_to_idx


def build_resnet18(
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """Build ResNet-18 model with custom linear classification head."""
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    in_features = model.fc.in_features
    if dropout > 0.0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    else:
        model.fc = nn.Linear(in_features, num_classes)

    return model


def set_backbone_trainable(model: nn.Module, trainable_layers: str = "head_only") -> None:
    """Configure parameter freezing for staged transfer learning.

    Options:
    - 'head_only': Freeze all layers except model.fc.
    - 'layer4_and_head': Freeze layers up to layer3; unfreeze layer4 and fc.
    - 'full': Unfreeze all parameters.
    """
    if trainable_layers == "head_only":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

    elif trainable_layers == "layer4_and_head":
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze layer4
        for param in model.layer4.parameters():
            param.requires_grad = True
        # Unfreeze classification head
        for param in model.fc.parameters():
            param.requires_grad = True

    elif trainable_layers == "full":
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown trainable_layers setting: {trainable_layers}")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    class_names: List[str],
    image_size: int,
    best_f1: float,
    training_config: Dict[str, Any],
    version: str = "1.0.0",
) -> None:
    """Save full checkpoint conforming to FarmWatch checkpoint contract."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {cls: CLASS_METADATA.get(cls, {"display_name": cls, "category_type": "unknown"}) for cls in class_names}

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "architecture": "resnet18",
        "class_names": class_names,
        "class_metadata": metadata,
        "image_size": [image_size, image_size],
        "normalization_mean": IMAGENET_MEAN,
        "normalization_std": IMAGENET_STD,
        "model_version": version,
        "best_validation_macro_f1": float(best_f1),
        "training_config": training_config,
        "source_dataset_reference": "Zenodo Record 19108517 (Subset)",
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path | str, device: torch.device
) -> Tuple[nn.Module, List[str], Dict[str, Any]]:
    """Load and validate FarmWatch checkpoint contract strictly."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    required_keys = [
        "model_state_dict",
        "architecture",
        "class_names",
        "class_metadata",
        "image_size",
        "normalization_mean",
        "normalization_std",
    ]
    for k in required_keys:
        if k not in checkpoint:
            raise KeyError(f"Checkpoint missing required key: '{k}'")

    class_names = checkpoint["class_names"]
    if not isinstance(class_names, list) or len(class_names) == 0:
        raise ValueError("Checkpoint class_names must be a non-empty list")

    arch = checkpoint["architecture"]
    if arch != "resnet18":
        raise ValueError(f"Unsupported architecture: '{arch}'. Only 'resnet18' supported in MVP.")

    has_sequential_head = any(k.startswith("fc.1.") for k in checkpoint["model_state_dict"].keys())
    dropout_val = 0.2 if has_sequential_head else 0.0
    model = build_resnet18(num_classes=len(class_names), pretrained=False, dropout=dropout_val)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    return model, class_names, checkpoint
