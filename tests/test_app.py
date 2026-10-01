"""
Unit tests for the FastAPI serving app (app/main.py).

The model loader is mocked in every test (patching app.main.load_serving_model,
called once during the app's lifespan) so these run without a real MLflow server,
a GPU, or any network access.
"""

import io
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


def _png_bytes(color=(128, 128, 128), size=(32, 32)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="PNG")
    return buf.getvalue()


def _sample_result():
    return {
        "predicted_class_id": 1,
        "predicted_label": "PNEUMONIA",
        "confidence": 0.87,
        "probabilities": {"NORMAL": 0.13, "PNEUMONIA": 0.87},
    }


@contextmanager
def client_with_model():
    """TestClient whose lifespan loads a mocked predictor that always returns a fixed result."""
    mock_predictor = MagicMock()
    mock_predictor.predict_image.return_value = _sample_result()
    with patch("app.main.load_serving_model", return_value=(mock_predictor, "pneumonia-classifier@champion (v1, densenet201)")):
        with TestClient(app) as client:
            yield client, mock_predictor


@contextmanager
def client_without_model():
    """TestClient whose lifespan finds no model at all (registry empty, no local weights)."""
    with patch("app.main.load_serving_model", return_value=(None, None)):
        with TestClient(app) as client:
            yield client


# ---------------------------------------------------------------------------
# health / root
# ---------------------------------------------------------------------------

def test_health_reports_loaded_model():
    with client_with_model() as (client, _predictor):
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == "pneumonia-classifier@champion (v1, densenet201)"


def test_health_reports_no_model_loaded():
    with client_without_model() as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "no_model_loaded"
    assert body["model_loaded"] is False
    assert body["model_version"] is None


def test_root_matches_health():
    with client_with_model() as (client, _predictor):
        root_resp = client.get("/")
        health_resp = client.get("/health")
    assert root_resp.json() == health_resp.json()


def test_metrics_endpoint_is_prometheus_format():
    with client_with_model() as (client, _predictor):
        resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "python_gc_objects_collected_total" in resp.text


def test_metrics_counts_predictions_by_label():
    with client_with_model() as (client, _predictor):
        client.post("/predict", files={"file": ("xray.png", _png_bytes(), "image/png")})
        resp = client.get("/metrics")
    assert 'pneumonia_predictions_total{predicted_label="PNEUMONIA"}' in resp.text


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------

def test_predict_returns_503_when_no_model_loaded():
    with client_without_model() as client:
        resp = client.post("/predict", files={"file": ("x.png", _png_bytes(), "image/png")})
    assert resp.status_code == 503


def test_predict_returns_400_for_invalid_image():
    with client_with_model() as (client, _predictor):
        resp = client.post("/predict", files={"file": ("not_an_image.txt", b"this is not image data", "text/plain")})
    assert resp.status_code == 400


def test_predict_returns_prediction_for_valid_image():
    with client_with_model() as (client, mock_predictor):
        resp = client.post("/predict", files={"file": ("xray.png", _png_bytes(), "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["filename"] == "xray.png"
    assert body["predicted_label"] == "PNEUMONIA"
    assert body["predicted_class_id"] == 1
    assert body["confidence"] == pytest.approx(0.87)
    assert body["probabilities"] == {"NORMAL": 0.13, "PNEUMONIA": 0.87}
    assert body["model_version"] == "pneumonia-classifier@champion (v1, densenet201)"
    mock_predictor.predict_image.assert_called_once()


def test_predict_passes_a_pil_image_to_the_predictor():
    """The route must decode the upload before handing it to predict_image (not raw bytes)."""
    with client_with_model() as (client, mock_predictor):
        client.post("/predict", files={"file": ("xray.png", _png_bytes(), "image/png")})
    called_with = mock_predictor.predict_image.call_args[0][0]
    assert isinstance(called_with, Image.Image)


# ---------------------------------------------------------------------------
# /predict/batch
# ---------------------------------------------------------------------------

def test_predict_batch_returns_503_when_no_model_loaded():
    with client_without_model() as client:
        resp = client.post("/predict/batch", files=[("files", ("a.png", _png_bytes(), "image/png"))])
    assert resp.status_code == 503


def test_predict_batch_returns_one_result_per_file():
    with client_with_model() as (client, mock_predictor):
        resp = client.post(
            "/predict/batch",
            files=[
                ("files", ("a.png", _png_bytes(), "image/png")),
                ("files", ("b.png", _png_bytes(color=(0, 0, 0)), "image/png")),
            ],
        )
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 2
    assert {r["filename"] for r in results} == {"a.png", "b.png"}
    assert mock_predictor.predict_image.call_count == 2


def test_predict_batch_fails_whole_request_on_one_bad_image():
    """Current behavior: a single invalid file in a batch aborts the whole request with 400."""
    with client_with_model() as (client, _predictor):
        resp = client.post(
            "/predict/batch",
            files=[
                ("files", ("a.png", _png_bytes(), "image/png")),
                ("files", ("bad.txt", b"not an image", "text/plain")),
            ],
        )
    assert resp.status_code == 400
