"""Tests for FarmWatch FastAPI backend endpoints and upload handling."""

from __future__ import annotations

import io
from pathlib import Path
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.model_service import model_service


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "FarmWatch API"
    assert "version" in data
    assert data["documentation"] == "/docs"


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["classes_count"] == 5


def test_classes_endpoint(client):
    response = client.get("/api/classes")
    assert response.status_code == 200
    data = response.json()
    assert "classes" in data
    assert len(data["classes"]) == 5
    canonical_names = [c["canonical_name"] for c in data["classes"]]
    assert "healthy" in canonical_names
    assert "low_stress" in canonical_names
    assert "high_stress" in canonical_names
    assert "rust" in canonical_names
    assert "bare_soil" in canonical_names


def test_predict_valid_image(client):
    # Create valid test image
    img = Image.new("RGB", (128, 128), color=(40, 180, 40))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/api/predict",
        files={"file": ("test.jpg", buf, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert "display_name" in data
    assert "category_type" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert len(data["probabilities"]) in (5, 9)
    # Check that probabilities sum to ~1.0
    prob_sum = sum(data["probabilities"].values())
    assert 0.99 <= prob_sum <= 1.01


def test_predict_corrupt_file(client):
    buf = io.BytesIO(b"MALFORMED_RANDOM_DATA_NOT_AN_IMAGE")
    response = client.post(
        "/api/predict",
        files={"file": ("fake.png", buf, "image/png")},
    )
    assert response.status_code == 400
    assert "Invalid or unreadable image" in response.json()["detail"]


def test_predict_unsupported_mime(client):
    buf = io.BytesIO(b"hello world")
    response = client.post(
        "/api/predict",
        files={"file": ("doc.pdf", buf, "application/pdf")},
    )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_predict_missing_file(client):
    response = client.post("/api/predict")
    assert response.status_code == 422
