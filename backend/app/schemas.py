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


class ClassesResponse(BaseModel):
    classes: List[ClassInfo]


class PredictionResponse(BaseModel):
    predicted_class: str = Field(..., description="Top canonical class prediction")
    display_name: str = Field(..., description="Human-friendly display name")
    category_type: str = Field(..., description="Category group (e.g. water_stress, disease, plant_condition, background)")
    confidence: float = Field(..., description="Top softmax confidence score (0.0 - 1.0)")
    confidence_warning: bool = Field(..., description="Whether confidence falls below threshold")
    probabilities: Dict[str, float] = Field(..., description="Probabilities for each class in checkpoint order")
    interpretation: str = Field(..., description="Cautious agronomic explanation of the result")
    model_version: str = Field(..., description="Model artifact version")
    notice: str = Field(
        default="AI image classification estimate based on maize training dataset. Not a confirmed agricultural diagnosis.",
        description="Standard medical/agricultural disclaimer",
    )
