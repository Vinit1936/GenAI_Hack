"""FarmWatch Backend Configuration Settings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "FarmWatch API"
    app_version: str = "1.0.0"
    model_path: Path = Path(os.getenv("FARMWATCH_MODEL_PATH", "artifacts/models/farmwatch_resnet18.pth"))
    frontend_origins: str = os.getenv("FRONTEND_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "10"))
    max_image_pixels: int = int(os.getenv("MAX_IMAGE_PIXELS", "89478485"))  # Pillow default bomb threshold
    confidence_warning_threshold: float = float(os.getenv("CONFIDENCE_WARNING_THRESHOLD", "0.60"))
    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", "8000"))

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def allowed_origins(self) -> List[str]:
        return [origin.strip() for origin in self.frontend_origins.split(",") if origin.strip()]


settings = Settings()
