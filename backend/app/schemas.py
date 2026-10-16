"""FarmWatch API Pydantic Request & Response Schemas."""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class RootResponse(BaseModel):
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="API Version")
    status: str = Field(..., description="Operational status")
    documentation: str = Field(..., description="Link to Swagger docs")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status ('healthy' or 'degraded')")
    model_loaded: bool = Field(..., description="Whether PyTorch model checkpoint is loaded in memory")
    model_architecture: Optional[str] = Field(None, description="Active architecture")
    device: Optional[str] = Field(None, description="Inference device")
    classes_count: int = Field(..., description="Number of supported classes")


class ClassInfo(BaseModel):
    canonical_name: str
    display_name: str
    category_type: str
    description: str
    disease_name: Optional[str] = None
    scientific_name: Optional[str] = None
    pathogen_type: Optional[str] = None
    host_crop: Optional[str] = "Maize / Corn (Zea mays)"
    symptoms: Optional[str] = None


class ClassesResponse(BaseModel):
    classes: List[ClassInfo]


class PredictionResponse(BaseModel):
    predicted_class: str = Field(..., description="Top canonical class prediction")
    display_name: str = Field(..., description="Human-friendly display name")
    category_type: str = Field(..., description="Category group (e.g. water_stress, disease, plant_condition, background)")
    disease_name: Optional[str] = Field(None, description="Actual common name of the plant disease or condition")
    scientific_name: Optional[str] = Field(None, description="Biological/scientific pathogen name (e.g. Puccinia sorghi)")
    pathogen_type: Optional[str] = Field(None, description="Pathogen type (Fungal, Abiotic, etc.)")
    host_crop: Optional[str] = Field("Maize / Corn (Zea mays)", description="Host crop species on which the model was trained")
    symptoms: Optional[str] = Field(None, description="Diagnostic visual symptoms associated with this condition")
    confidence: float = Field(..., description="Top softmax confidence score (0.0 - 1.0)")
    confidence_warning: bool = Field(..., description="Whether confidence falls below threshold")
    probabilities: Dict[str, float] = Field(..., description="Probabilities for each class in checkpoint order")
    interpretation: str = Field(..., description="Cautious agronomic explanation of the result")
    model_version: str = Field(..., description="Model artifact version")
    gradcam_image: Optional[str] = Field(None, description="Base64 encoded Grad-CAM attribution overlay image")
    notice: str = Field(
        default="AI image classification estimate based on maize training dataset. Not a confirmed agricultural diagnosis.",
        description="Standard medical/agricultural disclaimer",
    )
