"""FarmWatch FastAPI Application.

Provides REST API endpoints for crop condition classification,
health monitoring, and class metadata.
"""

from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from backend.app.config import settings
from backend.app.model_service import model_service
from backend.app.schemas import (
    ClassesResponse,
    ClassInfo,
    HealthResponse,
    PredictionResponse,
    RootResponse,
)
from training.model_utils import CLASS_METADATA

logger = logging.getLogger("farmwatch")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager: loads model checkpoint once during startup."""
    logger.info("Initializing FarmWatch application...")
    try:
        model_service.load()
        logger.info(f"Model checkpoint successfully loaded from '{settings.model_path}'.")
    except FileNotFoundError as e:
        logger.warning(
            f"Model checkpoint not found during startup: {e}. "
            "Server running in degraded mode. Train model with: 'python -m training.train'"
        )
    except Exception as e:
        logger.error(f"Failed to load model checkpoint: {e}", exc_info=True)

    yield

    logger.info("Shutting down FarmWatch application...")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Field Crop Water Stress and Visible Condition Classifier API",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/", response_model=RootResponse)
async def root() -> RootResponse:
    """Service root endpoint pointing to documentation."""
    return RootResponse(
        service=settings.app_name,
        version=settings.app_version,
        status="healthy" if model_service.is_loaded else "degraded",
        documentation="/docs",
    )


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Service health and model availability check."""
    return HealthResponse(
        status="healthy" if model_service.is_loaded else "degraded",
        model_loaded=model_service.is_loaded,
        model_architecture=str(model_service.checkpoint_metadata.get("architecture")) if model_service.is_loaded else None,
        device=str(model_service.device),
        classes_count=len(model_service.class_names),
    )


@app.get("/api/classes", response_model=ClassesResponse)
async def list_classes() -> ClassesResponse:
    """Returns canonical classes, display names, and category descriptions."""
    classes = []
    # If model is loaded, use exact checkpoint classes order; otherwise use default metadata order
    class_keys = model_service.class_names if model_service.is_loaded else list(CLASS_METADATA.keys())

    for k in class_keys:
        meta = CLASS_METADATA.get(k, {"display_name": k, "category_type": "unknown", "description": "Class category"})
        classes.append(ClassInfo(
            canonical_name=k,
            display_name=meta["display_name"],
            category_type=meta["category_type"],
            description=meta["description"],
        ))

    return ClassesResponse(classes=classes)


@app.post(
    "/api/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"description": "Invalid image or size exceeded"},
        503: {"description": "Model checkpoint not loaded"},
    },
)
async def predict_image(file: UploadFile = File(..., description="Uploaded image file (JPEG or PNG)")) -> PredictionResponse:
    """Classifies an uploaded crop photograph using the loaded PyTorch model."""
    if not model_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready for inference. Please ensure model checkpoint exists.",
        )

    # Basic MIME validation
    allowed_content_types = {"image/jpeg", "image/png", "image/webp", "image/tiff"}
    if file.content_type and file.content_type.lower() not in allowed_content_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{file.content_type}'. Only JPEG, PNG, and TIFF formats are accepted.",
        )

    # Bounded read to avoid memory exhaustion
    max_bytes = settings.max_upload_mb * 1024 * 1024
    content = await file.read(max_bytes + 1024)
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file exceeds {settings.max_upload_mb} MB limit.",
        )

    try:
        result = model_service.predict(content)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the image. Please verify file integrity.",
        )


@app.get("/api/sample/{class_name}")
async def get_sample_image(class_name: str) -> Response:
    """Provides a sample image from the held-out test split as standard JPEG for rapid verification."""
    clean_name = class_name.lower().strip()
    sample_dir = Path(f"data/processed/test/{clean_name}")
    if not sample_dir.is_dir():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No samples for class '{clean_name}'.")

    files = sorted(list(sample_dir.glob("*.tif")))
    if not files:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No test files found for '{clean_name}'.")

    # Convert TIFF patch to standard JPEG for universal browser compatibility
    with Image.open(files[0]) as img:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=95)
        jpeg_bytes = buf.getvalue()

    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={"Content-Disposition": f'inline; filename="sample_{clean_name}.jpg"'},
    )
