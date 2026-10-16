"""Grad-CAM (Gradient-weighted Class Activation Mapping) for FarmWatch ResNet-18.

Generates visual attribution heatmaps highlighting the spatial regions of crop foliage
that contributed most strongly to the model's classification decision.

Notice: Grad-CAM produces visual attribution maps for model interpretability and debugging;
it does not constitute a biological or pathology-certified lesion map.
"""

from __future__ import annotations

import base64
import io
from typing import Optional, Tuple

import matplotlib.cm as cm
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from training.model_utils import DEFAULT_IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


class GradCAM:
    """Computes Grad-CAM attribution heatmaps for convolutional architectures."""

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()

        # Default to the final convolutional residual block of ResNet-18
        if target_layer is None:
            if hasattr(model, "layer4"):
                self.target_layer = model.layer4[-1]
            else:
                raise ValueError("Target layer must be specified for non-ResNet architectures.")
        else:
            self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Computes 2D normalized Grad-CAM heatmap array [0.0, 1.0]."""
        self.model.zero_grad()

        # Forward pass (enable grad for backward hook)
        with torch.enable_grad():
            tensor = input_tensor.clone().requires_grad_(True)
            logits = self.model(tensor)

            if target_class_idx is None:
                target_class_idx = int(torch.argmax(logits, dim=1).item())

            score = logits[0, target_class_idx]
            score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed to capture gradients or activations.")

        # Global average pooling of gradients: shape (Channels,)
        weights = torch.mean(self.gradients[0], dim=(1, 2))

        # Linear combination of channels: shape (H, W)
        cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32, device=self.activations.device)
        for i, w in enumerate(weights):
            cam += w * self.activations[0, i]

        # ReLU to focus on positive contributions
        cam = F.relu(cam)

        cam_np = cam.cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np


def generate_gradcam_overlay(
    model: nn.Module,
    pil_image: Image.Image,
    target_class_idx: Optional[int] = None,
    alpha: float = 0.55,
) -> Tuple[str, np.ndarray]:
    """Generates a Grad-CAM overlay image and returns base64 data URL and raw heatmap.

    Returns:
        (base64_data_url, heatmap_array)
    """
    device = next(model.parameters()).device
    transform = transforms.Compose([
        transforms.Resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    input_tensor = transform(pil_image.convert("RGB")).unsqueeze(0).to(device)

    grad_cam = GradCAM(model)
    try:
        heatmap = grad_cam.generate_heatmap(input_tensor, target_class_idx=target_class_idx)
    finally:
        grad_cam.remove_hooks()

    # Resize original image to standard dimensions
    resized_orig = pil_image.convert("RGB").resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))
    orig_np = np.array(resized_orig, dtype=np.float32) / 255.0

    # Upscale heatmap to match input resolution
    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
        (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE), resample=Image.Resampling.BILINEAR
    )
    heatmap_scaled = np.array(heatmap_img, dtype=np.float32) / 255.0

    # Apply turbo/jet colormap
    import matplotlib
    colormap = matplotlib.colormaps["jet"]
    colored_heatmap = colormap(heatmap_scaled)[:, :, :3]  # Drop alpha

    # Blend original image and heatmap
    blended = (1.0 - alpha) * orig_np + alpha * colored_heatmap
    blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)

    blended_pil = Image.fromarray(blended)

    # Encode to base64 JPEG
    buffer = io.BytesIO()
    blended_pil.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{encoded}"

    return data_url, heatmap
