"""FarmWatch Model Inference Service.

Handles checkpoint loading at startup, deterministic image preprocessing,
and inference execution in torch.inference_mode().
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageOps
import torch
import torchvision.transforms as transforms

from backend.app.config import settings
from backend.app.schemas import PredictionResponse
from training.model_utils import (
    CLASS_METADATA,
    DEFAULT_IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    load_checkpoint,
)


from training.train_solanaceae import SOLANACEAE_METADATA


class ModelService:
    """Encapsulates PyTorch model lifecycle and inference for multiple crop specialists."""

    def __init__(self, device_name: str = "auto"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else torch.device(device_name)
        self.models: Dict[str, torch.nn.Module] = {}
        self.classes_by_crop: Dict[str, List[str]] = {}
        self.metadata_by_crop: Dict[str, Dict[str, Any]] = {}
        self.is_loaded: bool = False

        self.transform = transforms.Compose([
            transforms.Resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    @property
    def class_names(self) -> List[str]:
        """Provides backward-compatible list of baseline (maize) class names."""
        return self.classes_by_crop.get("maize", [])

    @property
    def checkpoint_metadata(self) -> Dict[str, Any]:
        """Provides backward-compatible metadata dict of baseline (maize) model."""
        return self.metadata_by_crop.get("maize", {})

    def load(self) -> None:
        """Loads both Maize and Solanaceae model checkpoints into memory."""
        # 1. Load Maize Checkpoint (Baseline)
        if settings.model_path.is_file():
            model_maize, classes_maize, meta_maize = load_checkpoint(settings.model_path, self.device)
            self.models["maize"] = model_maize
            self.classes_by_crop["maize"] = classes_maize
            self.metadata_by_crop["maize"] = meta_maize
            print(f"[INFO] ModelService: Maize checkpoint loaded ({len(classes_maize)} classes)")
        else:
            print(f"[WARN] ModelService: Maize checkpoint not found at {settings.model_path}")

        # 2. Load Solanaceae Checkpoint (Potato & Tomato Specialist)
        if settings.solanaceae_model_path.is_file():
            model_sol, classes_sol, meta_sol = load_checkpoint(settings.solanaceae_model_path, self.device)
            self.models["solanaceae"] = model_sol
            self.classes_by_crop["solanaceae"] = classes_sol
            self.metadata_by_crop["solanaceae"] = meta_sol
            print(f"[INFO] ModelService: Solanaceae (Potato & Tomato) checkpoint loaded ({len(classes_sol)} classes)")
        else:
            print(f"[WARN] ModelService: Solanaceae checkpoint not found at {settings.solanaceae_model_path}")

        self.is_loaded = len(self.models) > 0

    def resolve_crop_key(self, crop: str) -> str:
        """Resolves user crop parameter into canonical domain ('maize', 'solanaceae', or 'auto')."""
        c = (crop or "").strip().lower()
        if c in ["auto", "unified", "all", "detect", ""]:
            return "auto"
        if c in ["solanaceae", "potato", "tomato", "potatoes", "tomatoes"]:
            return "solanaceae"
        return "maize"

    def get_classes(self, crop: str = "maize") -> List[str]:
        """Returns class names for requested crop domain."""
        crop_key = self.resolve_crop_key(crop)
        if crop_key in self.classes_by_crop:
            return self.classes_by_crop[crop_key]
        return list(CLASS_METADATA.keys()) if crop_key == "maize" else list(SOLANACEAE_METADATA.keys())

    def get_metadata_catalog(self, crop: str = "maize") -> Dict[str, Any]:
        """Returns metadata definitions for requested crop domain."""
        crop_key = self.resolve_crop_key(crop)
        return SOLANACEAE_METADATA if crop_key == "solanaceae" else CLASS_METADATA

    def validate_and_load_image(self, image_bytes: bytes) -> Image.Image:
        """Validates raw image bytes and returns a sanitized RGB Pillow Image."""
        if not image_bytes:
            raise ValueError("Uploaded image file is empty.")

        if len(image_bytes) > settings.max_upload_mb * 1024 * 1024:
            raise ValueError(f"Image exceeds maximum upload limit of {settings.max_upload_mb} MB.")

        try:
            Image.MAX_IMAGE_PIXELS = settings.max_image_pixels
            with Image.open(io.BytesIO(image_bytes)) as img:
                img.verify()

            img = Image.open(io.BytesIO(image_bytes))
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")

            if img.width == 0 or img.height == 0:
                raise ValueError("Image dimensions must be non-zero.")

            return img
        except Exception as e:
            raise ValueError(f"Invalid or unreadable image file: {str(e)}")

    def predict(self, image_bytes: bytes, crop: str = "auto") -> PredictionResponse:
        """Runs full prediction pipeline on uploaded image bytes using auto-routing or explicit crop specialist."""
        if not self.is_loaded:
            raise RuntimeError("Model service is not loaded.")

        crop_key = self.resolve_crop_key(crop)

        # 1. Image preprocessing
        pil_image = self.validate_and_load_image(image_bytes)
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        if crop_key == "auto":
            # Auto-routing across available specialists
            candidates: Dict[str, Dict[str, Any]] = {}
            for dom in ["maize", "solanaceae"]:
                if dom not in self.models:
                    continue
                model = self.models[dom]
                classes = self.classes_by_crop[dom]
                with torch.inference_mode():
                    logits = model(tensor)
                    probs = torch.softmax(logits, dim=1).squeeze(0)
                probs_list = probs.cpu().numpy().tolist()
                top_idx = int(torch.argmax(probs).item())
                top_class = classes[top_idx]
                conf = float(probs_list[top_idx])
                candidates[dom] = {
                    "top_idx": top_idx,
                    "top_class": top_class,
                    "confidence": conf,
                    "probs_list": probs_list,
                    "classes": classes,
                }

            # Decision logic: prioritize foliage condition/disease over bare soil background
            if "maize" in candidates and "solanaceae" in candidates:
                m = candidates["maize"]
                s = candidates["solanaceae"]
                if m["top_class"] == "bare_soil" and s["top_class"] == "bare_soil":
                    crop_key = "maize"
                elif m["top_class"] == "bare_soil" and s["top_class"] != "bare_soil":
                    crop_key = "solanaceae"
                elif s["top_class"] == "bare_soil" and m["top_class"] != "bare_soil":
                    crop_key = "maize"
                else:
                    crop_key = "solanaceae" if s["confidence"] >= m["confidence"] else "maize"
            elif "solanaceae" in candidates:
                crop_key = "solanaceae"
            else:
                crop_key = "maize"

        if crop_key not in self.models:
            # Fallback to available model if requested domain is not loaded
            fallback_key = "maize" if "maize" in self.models else next(iter(self.models.keys()))
            crop_key = fallback_key

        active_model = self.models[crop_key]
        active_classes = self.classes_by_crop[crop_key]
        active_metadata = SOLANACEAE_METADATA if crop_key == "solanaceae" else CLASS_METADATA

        # 2. PyTorch Inference Mode
        with torch.inference_mode():
            logits = active_model(tensor)
            probs_tensor = torch.softmax(logits, dim=1).squeeze(0)

        probs_list = probs_tensor.cpu().numpy().tolist()
        top_idx = int(torch.argmax(probs_tensor).item())
        top_class = active_classes[top_idx]
        confidence = float(probs_list[top_idx])

        # 3. Class probabilities mapping
        probabilities = {cls: round(float(probs_list[i]), 4) for i, cls in enumerate(active_classes)}

        meta = active_metadata.get(top_class, {
            "display_name": top_class,
            "category_type": "unknown",
            "description": "Visible condition detected by model.",
        })

        is_warning = confidence < settings.confidence_warning_threshold

        host_crop_name = meta.get("host_crop") or ("Maize / Corn (Zea mays)" if crop_key == "maize" else "Nightshade (Solanaceae)")
        interpretation = (
            f"The image resembles the '{meta['display_name']}' class on {host_crop_name} "
            f"with {confidence:.1%} estimated confidence. "
        )
        if is_warning:
            interpretation += "Note: Confidence score is below threshold; treat with added caution or re-take photo."

        # 4. Optional Grad-CAM Attribution Overlay
        gradcam_b64 = None
        try:
            from training.gradcam import generate_gradcam_overlay
            gradcam_b64, _ = generate_gradcam_overlay(active_model, pil_image, target_class_idx=top_idx)
        except Exception:
            pass

        return PredictionResponse(
            predicted_class=top_class,
            display_name=meta["display_name"],
            category_type=meta["category_type"],
            disease_name=meta.get("disease_name"),
            scientific_name=meta.get("scientific_name"),
            pathogen_type=meta.get("pathogen_type"),
            host_crop=host_crop_name,
            symptoms=meta.get("symptoms"),
            confidence=round(confidence, 4),
            confidence_warning=is_warning,
            probabilities=probabilities,
            interpretation=interpretation,
            model_version=str(self.metadata_by_crop.get(crop_key, {}).get("model_version", "1.0.0")),
            gradcam_image=gradcam_b64,
        )


# Singleton instance for application lifecycle
model_service = ModelService()
