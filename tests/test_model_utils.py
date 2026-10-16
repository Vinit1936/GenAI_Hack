"""Tests for FarmWatch model creation, transforms, and checkpoint contracts."""

from __future__ import annotations

import tempfile
from pathlib import Path
import torch
import pytest
from PIL import Image

from training.model_utils import (
    DEFAULT_IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_resnet18,
    get_transforms,
    load_checkpoint,
    save_checkpoint,
    set_backbone_trainable,
)


def test_model_factory_head():
    model = build_resnet18(num_classes=5, pretrained=False, dropout=0.2)
    dummy_input = torch.randn(2, 3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
    output = model(dummy_input)
    assert output.shape == (2, 5)
    assert not torch.isnan(output).any()


def test_freezing_modes():
    model = build_resnet18(num_classes=5, pretrained=False, dropout=0.2)

    # Head only
    set_backbone_trainable(model, "head_only")
    for name, param in model.named_parameters():
        if "fc" in name:
            assert param.requires_grad is True
        else:
            assert param.requires_grad is False

    # Layer4 and head
    set_backbone_trainable(model, "layer4_and_head")
    for name, param in model.named_parameters():
        if "fc" in name or "layer4" in name:
            assert param.requires_grad is True
        else:
            assert param.requires_grad is False


def test_transforms_shape_and_norm():
    train_tf, eval_tf = get_transforms(DEFAULT_IMAGE_SIZE)
    img = Image.new("RGB", (100, 80), color=(128, 128, 128))
    tensor = eval_tf(img)
    assert tensor.shape == (3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
    assert tensor.dtype == torch.float32


def test_checkpoint_round_trip(tmp_path):
    ckpt_path = tmp_path / "test_model.pth"
    class_names = ["bare_soil", "healthy", "high_stress", "low_stress", "rust"]
    model = build_resnet18(num_classes=5, pretrained=False, dropout=0.2)

    save_checkpoint(
        path=ckpt_path,
        model=model,
        class_names=class_names,
        image_size=DEFAULT_IMAGE_SIZE,
        best_f1=0.88,
        training_config={"epochs": 2},
        version="0.1.0",
    )

    assert ckpt_path.is_file()

    loaded_model, loaded_classes, meta = load_checkpoint(ckpt_path, torch.device("cpu"))
    assert loaded_classes == class_names
    assert meta["best_validation_macro_f1"] == 0.88
    assert meta["model_version"] == "0.1.0"
    assert meta["architecture"] == "resnet18"
