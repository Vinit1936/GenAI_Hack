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


class ModelService:
    """Encapsulates PyTorch model lifecycle and inference."""

    def __init__(self, model_path: Optional[Path] = None, device_name: str = "auto"):
        self.model_path = model_path or settings.model_path
        if device_name == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_name)

        self.model: Optional[torch.nn.Module] = None
        self.class_names: List[str] = []
        self.checkpoint_metadata: Dict[str, Any] = {}
        self.is_loaded: bool = False

        self.transform = transforms.Compose([
            transforms.Resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def load(self) -> None:
        """Loads model checkpoint into memory strictly."""
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found at '{self.model_path}'. "
                "Ensure training pipeline has executed: 'python -m training.train'"
            )

        model, class_names, metadata = load_checkpoint(self.model_path, self.device)
        self.model = model
        self.class_names = class_names
        self.checkpoint_metadata = metadata
        self.is_loaded = True
        print(f"[INFO] ModelService: Checkpoint loaded successfully from {self.model_path} ({len(self.class_names)} classes)")

    def validate_and_load_image(self, image_bytes: bytes) -> Image.Image:
        """Validates raw image bytes and returns a sanitized RGB Pillow Image."""
        if not image_bytes:
            raise ValueError("Uploaded image file is empty.")

        if len(image_bytes) > settings.max_upload_mb * 1024 * 1024:
            raise ValueError(f"Image exceeds maximum upload limit of {settings.max_upload_mb} MB.")

        try:
            # Check decompression bomb threshold
            Image.MAX_IMAGE_PIXELS = settings.max_image_pixels
            with Image.open(io.BytesIO(image_bytes)) as img:
                img.verify()

            # Re-open for actual processing and EXIF handling
            img = Image.open(io.BytesIO(image_bytes))
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")

            if img.width == 0 or img.height == 0:
                raise ValueError("Image dimensions must be non-zero.")

            return img
        except Exception as e:
            raise ValueError(f"Invalid or unreadable image file: {str(e)}")

    def predict(self, image_bytes: bytes) -> PredictionResponse:
        """Runs full prediction pipeline on uploaded image bytes."""
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model service is not loaded.")

        # 1. Image preprocessing
        pil_image = self.validate_and_load_image(image_bytes)
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        # 2. PyTorch Inference Mode
        with torch.inference_mode():
            logits = self.model(tensor)
            probs_tensor = torch.softmax(logits, dim=1).squeeze(0)

        probs_list = probs_tensor.cpu().numpy().tolist()
        top_idx = int(torch.argmax(probs_tensor).item())
        top_class = self.class_names[top_idx]
        confidence = float(probs_list[top_idx])

        # 3. Class probabilities mapping in exact checkpoint order
        probabilities = {cls: round(float(probs_list[i]), 4) for i, cls in enumerate(self.class_names)}

        meta = CLASS_METADATA.get(top_class, {
            "display_name": top_class,
            "category_type": "unknown",
            "description": "Visible condition detected by model.",
        })

        is_warning = confidence < settings.confidence_warning_threshold

        interpretation = (
            f"The image resembles the '{meta['display_name']}' class in the training dataset "
            f"with {confidence:.1%} estimated confidence. "
        )
        if is_warning:
            interpretation += "Note: Confidence score is below threshold; treat with added caution or re-take photo."

        return PredictionResponse(
            predicted_class=top_class,
            display_name=meta["display_name"],
            category_type=meta["category_type"],
            disease_name=meta.get("disease_name"),
            scientific_name=meta.get("scientific_name"),
            pathogen_type=meta.get("pathogen_type"),
            host_crop=meta.get("host_crop", "Maize / Corn (Zea mays)"),
            symptoms=meta.get("symptoms"),
            confidence=round(confidence, 4),
            confidence_warning=is_warning,
            probabilities=probabilities,
            interpretation=interpretation,
            model_version=str(self.checkpoint_metadata.get("model_version", "1.0.0")),
        )


# Singleton instance for application lifecycle
model_service = ModelService()
