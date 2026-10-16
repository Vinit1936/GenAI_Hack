"""Potato and Tomato (Solanaceae) Specialist ResNet-18 Training Pipeline.

Trains transfer-learning model on the 9-class Potato, Tomato, and Soil dataset:
  - potato_early_blight
  - potato_late_blight
  - potato_healthy
  - tomato_bacterial_spot
  - tomato_early_blight
  - tomato_late_blight
  - tomato_yellow_leaf_curl
  - tomato_healthy
  - bare_soil
Saves best checkpoint by validation Macro F1 to artifacts/models/farmwatch_solanaceae_resnet18.pth.
"""

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

from training.model_utils import (
    DEFAULT_IMAGE_SIZE,
    FarmWatchDataset,
    get_transforms,
    load_split_records,
    save_checkpoint,
    set_backbone_trainable,
)

SOLANACEAE_METADATA = {
    "potato_early_blight": {
        "display_name": "Potato Early Blight (Alternaria solani)",
        "category_type": "disease",
        "disease_name": "Potato Early Blight",
        "scientific_name": "Alternaria solani",
        "pathogen_type": "Fungal Pathogen (Ascomycota)",
        "host_crop": "Potato (Solanum tuberosum)",
        "symptoms": "Dark brown to black necrotic spots with characteristic concentric rings ('bullseye' target pattern), often surrounded by a chlorotic yellow halo.",
        "description": "Foliar fungal infection causing concentric target-spot lesions on potato foliage.",
    },
    "potato_late_blight": {
        "display_name": "Potato Late Blight (Phytophthora infestans)",
        "category_type": "disease",
        "disease_name": "Potato Late Blight",
        "scientific_name": "Phytophthora infestans",
        "pathogen_type": "Oomycete / Water Mold",
        "host_crop": "Potato (Solanum tuberosum)",
        "symptoms": "Water-soaked, dark brownish-black lesions with pale green borders, accompanied by delicate white fungal sporulation on leaf undersides in humid conditions.",
        "description": "Destructive oomycete disease causing rapid foliar necrosis and tuber rot.",
    },
    "potato_healthy": {
        "display_name": "Healthy Potato Foliage",
        "category_type": "plant_condition",
        "disease_name": "None (Healthy Crop)",
        "scientific_name": "N/A",
        "pathogen_type": "None",
        "host_crop": "Potato (Solanum tuberosum)",
        "symptoms": "Vigorous, uniform dark green compound leaves with intact leaflet edges and zero necrotic spotting.",
        "description": "Healthy potato canopy free from foliar blight or pest injury.",
    },
    "tomato_bacterial_spot": {
        "display_name": "Tomato Bacterial Spot (Xanthomonas spp.)",
        "category_type": "disease",
        "disease_name": "Tomato Bacterial Spot",
        "scientific_name": "Xanthomonas perforans / euvesicatoria",
        "pathogen_type": "Bacterial Pathogen (Gammaproteobacteria)",
        "host_crop": "Tomato (Solanum lycopersicum)",
        "symptoms": "Small, dark, circular to irregular water-soaked spots (1–3 mm) that turn dark brown to black, often with a conspicuous yellow halo.",
        "description": "Bacterial foliar disease causing greasy angular lesions and defoliation.",
    },
    "tomato_early_blight": {
        "display_name": "Tomato Early Blight (Alternaria solani)",
        "category_type": "disease",
        "disease_name": "Tomato Early Blight",
        "scientific_name": "Alternaria solani",
        "pathogen_type": "Fungal Pathogen (Ascomycota)",
        "host_crop": "Tomato (Solanum lycopersicum)",
        "symptoms": "Brown to black lesions with distinct concentric target rings on older lower leaves, causing progressive yellowing and defoliation.",
        "description": "Common foliar fungal blight characterized by concentric target lesions.",
    },
    "tomato_late_blight": {
        "display_name": "Tomato Late Blight (Phytophthora infestans)",
        "category_type": "disease",
        "disease_name": "Tomato Late Blight",
        "scientific_name": "Phytophthora infestans",
        "pathogen_type": "Oomycete / Water Mold",
        "host_crop": "Tomato (Solanum lycopersicum)",
        "symptoms": "Large, expanding, irregular water-soaked greasy gray-green to brown lesions on foliage and stems, with white downy growth on the undersides.",
        "description": "Devastating water-mold infection capable of destroying entire tomato canopies.",
    },
    "tomato_yellow_leaf_curl": {
        "display_name": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "category_type": "disease",
        "disease_name": "Tomato Yellow Leaf Curl",
        "scientific_name": "Tomato yellow leaf curl virus (TYLCV)",
        "pathogen_type": "Viral Pathogen (Begomovirus / Geminiviridae)",
        "host_crop": "Tomato (Solanum lycopersicum)",
        "symptoms": "Marked upward curling and cupping of leaflets, severe stunting of shoots, chlorotic yellow margins, and reduced leaflet size.",
        "description": "Whitefly-vectored plant virus causing severe foliar curling and stunted yield.",
    },
    "tomato_healthy": {
        "display_name": "Healthy Tomato Foliage",
        "category_type": "plant_condition",
        "disease_name": "None (Healthy Crop)",
        "scientific_name": "N/A",
        "pathogen_type": "None",
        "host_crop": "Tomato (Solanum lycopersicum)",
        "symptoms": "Deep green serrated leaflets with robust turgor pressure, clean veins, and zero mottling or spotting.",
        "description": "Healthy tomato foliage displaying normal vegetative vigor.",
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


def build_solanaceae_model(num_classes: int = 9, dropout: float = 0.2) -> nn.Module:
    """Builds ResNet-18 model with custom classification head for Solanaceae."""
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def train_solanaceae_specialist(
    splits_csv: Path = Path("data/processed_solanaceae/splits.csv"),
    data_dir: Path = Path("data/processed_solanaceae"),
    output_checkpoint: Path = Path("artifacts/models/farmwatch_solanaceae_resnet18.pth"),
    epochs_head: int = 4,
    epochs_fine: int = 5,
    batch_size: int = 16,
    lr: float = 1e-4,
    device_name: str = "auto",
) -> Dict[str, Any]:
    """Trains ResNet-18 model on Potato and Tomato dataset with early checkpointing."""
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print(f"[INFO] Initializing Solanaceae specialist training on {device}...")

    # Load splits
    train_recs, classes, class_to_idx = load_split_records(splits_csv, "train")
    val_recs, _, _ = load_split_records(splits_csv, "val")
    test_recs, _, _ = load_split_records(splits_csv, "test")

    print(f"[INFO] Loaded splits: {len(train_recs)} train, {len(val_recs)} val, {len(test_recs)} test across {len(classes)} classes.")

    train_tf, eval_tf = get_transforms(DEFAULT_IMAGE_SIZE)
    train_ds = FarmWatchDataset(data_dir, train_recs, class_to_idx, transform=train_tf)
    val_ds = FarmWatchDataset(data_dir, val_recs, class_to_idx, transform=eval_tf)
    test_ds = FarmWatchDataset(data_dir, test_recs, class_to_idx, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = build_solanaceae_model(num_classes=len(classes), dropout=0.2).to(device)
    criterion = nn.CrossEntropyLoss()

    # Stage 1: Train classification head only
    print(f"\n[STAGE 1] Training classification head for {epochs_head} epochs...")
    set_backbone_trainable(model, "head_only")
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-2)

    best_val_f1 = 0.0
    best_weights = None

    total_epochs = epochs_head + epochs_fine
    current_epoch = 0

    for epoch in range(1, epochs_head + 1):
        current_epoch += 1
        model.train()
        train_loss = 0.0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(imgs)

        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_targets.extend(targets.numpy())

        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
        print(f"Epoch [{current_epoch}/{total_epochs}] (Head) -> Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2%} | Val Macro F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = copy.deepcopy(model.state_dict())

    # Stage 2: Fine-tune layer4 + head
    print(f"\n[STAGE 2] Fine-tuning layer4 + head for {epochs_fine} epochs...")
    set_backbone_trainable(model, "layer4_and_head")
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-2)

    for epoch in range(1, epochs_fine + 1):
        current_epoch += 1
        model.train()
        train_loss = 0.0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(imgs)

        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_targets.extend(targets.numpy())

        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
        print(f"Epoch [{current_epoch}/{total_epochs}] (Fine) -> Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2%} | Val Macro F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = copy.deepcopy(model.state_dict())

    # Load best weights and evaluate on held-out test split
    print(f"\n[INFO] Restoring best model weights (Best Val Macro F1: {best_val_f1:.4f})...")
    if best_weights is not None:
        model.load_state_dict(best_weights)

    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            test_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            test_targets.extend(targets.numpy())

    test_acc = float(accuracy_score(test_targets, test_preds))
    test_f1 = float(f1_score(test_targets, test_preds, average="macro", zero_division=0))
    test_prec = float(precision_score(test_targets, test_preds, average="macro", zero_division=0))
    test_rec = float(recall_score(test_targets, test_preds, average="macro", zero_division=0))

    print(f"\n=== HELD-OUT TEST EVALUATION ({len(test_ds)} samples) ===")
    print(f"Test Accuracy:        {test_acc:.2%}")
    print(f"Test Macro Precision: {test_prec:.4f}")
    print(f"Test Macro Recall:    {test_rec:.4f}")
    print(f"Test Macro F1:        {test_f1:.4f}")

    # Save checkpoint
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "model_state_dict": model.state_dict(),
        "architecture": "resnet18",
        "num_classes": len(classes),
        "class_names": classes,
        "class_metadata": SOLANACEAE_METADATA,
        "image_size": [DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE],
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
        "best_validation_macro_f1": best_val_f1,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "model_version": "1.0.0",
        "training_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "crop_domain": "solanaceae_potato_tomato",
    }
    torch.save(checkpoint_payload, output_checkpoint)
    print(f"[SUCCESS] Checkpoint saved to '{output_checkpoint}' (Size: {output_checkpoint.stat().st_size / (1024*1024):.2f} MB)")

    return {
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "checkpoint": str(output_checkpoint),
    }


if __name__ == "__main__":
    train_solanaceae_specialist()
