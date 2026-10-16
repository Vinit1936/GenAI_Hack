"""Integration tests for Solanaceae (Potato & Tomato) model routing and diagnostics."""

from __future__ import annotations

import io
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_solanaceae_classes(client):
    response = client.get("/api/classes?crop=solanaceae")
    assert response.status_code == 200
    data = response.json()
    assert "classes" in data
    assert len(data["classes"]) == 9

    canonical_names = [c["canonical_name"] for c in data["classes"]]
    assert "potato_early_blight" in canonical_names
    assert "potato_late_blight" in canonical_names
    assert "potato_healthy" in canonical_names
    assert "tomato_early_blight" in canonical_names
    assert "tomato_late_blight" in canonical_names
    assert "tomato_bacterial_spot" in canonical_names
    assert "tomato_yellow_leaf_curl" in canonical_names
    assert "tomato_healthy" in canonical_names
    assert "bare_soil" in canonical_names


def test_solanaceae_predict(client):
    # Synthetic leaf image
    img = Image.new("RGB", (224, 224), color=(60, 160, 45))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/api/predict?crop=solanaceae",
        files={"file": ("leaf.jpg", buf, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert "display_name" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert len(data["probabilities"]) == 9
    assert 0.99 <= sum(data["probabilities"].values()) <= 1.01
    assert "Nightshade" in data.get("host_crop", "") or "Potato" in data.get("host_crop", "") or "Tomato" in data.get("host_crop", "")


def test_solanaceae_samples(client):
    # Test sample retrieval for potato and tomato
    resp_potato = client.get("/api/sample/potato_late_blight?crop=solanaceae")
    assert resp_potato.status_code == 200
    assert resp_potato.headers["content-type"] == "image/jpeg"
    assert len(resp_potato.content) > 1000

    resp_tomato = client.get("/api/sample/tomato_early_blight?crop=solanaceae")
    assert resp_tomato.status_code == 200
    assert resp_tomato.headers["content-type"] == "image/jpeg"
    assert len(resp_tomato.content) > 1000
