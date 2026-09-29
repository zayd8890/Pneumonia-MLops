"""
FastAPI serving app for the Pneumonia Classification model.

Endpoints:
    GET  /                health/version summary
    GET  /health           liveness/readiness probe (for Kubernetes later)
    POST /predict          single image -> label, confidence, probabilities
    POST /predict/batch    multiple images -> one result per image

Run locally:
    uvicorn app.main:app --host 0.0.0.0 --port 8000

The model is loaded once at startup (see app/model_loader.py), not per-request.
"""

import io
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from app.model_loader import load_serving_model

# Populated at startup; see lifespan() below.
state: Dict[str, object] = {"predictor": None, "model_version": None}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    predictor, version = load_serving_model()
    state["predictor"] = predictor
    state["model_version"] = version
    if predictor is None:
        print("[WARN] No model could be loaded (no MLflow champion, no local weights). "
              "/predict will return 503 until MODEL_WEIGHTS_PATH or the registry is available.")
    else:
        print(f"[INFO] Serving model: {version}")
    yield
    state["predictor"] = None


app = FastAPI(
    title="Pneumonia Classification API",
    description="Serves the MLflow-registered champion model for chest X-ray pneumonia classification.",
    version="1.0.0",
    lifespan=lifespan,
)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None


class PredictionResponse(BaseModel):
    filename: str
    predicted_label: str
    predicted_class_id: int
    confidence: float
    probabilities: Dict[str, float]
    model_version: str


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]


@app.get("/", response_model=HealthResponse)
def root() -> HealthResponse:
    return _health()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return _health()


def _health() -> HealthResponse:
    predictor = state.get("predictor")
    return HealthResponse(
        status="ok" if predictor is not None else "no_model_loaded",
        model_loaded=predictor is not None,
        model_version=state.get("model_version"),
    )


def _require_predictor():
    predictor = state.get("predictor")
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="No model loaded. Register a champion in MLflow or set MODEL_WEIGHTS_PATH.",
        )
    return predictor


async def _read_image(upload: UploadFile) -> Image.Image:
    raw = await upload.read()
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail=f"'{upload.filename}' is not a valid image file.") from None


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:  # noqa: B008 - FastAPI idiom
    predictor = _require_predictor()
    image = await _read_image(file)
    result = predictor.predict_image(image)
    return PredictionResponse(filename=file.filename, model_version=state["model_version"], **result)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(files: List[UploadFile] = File(...)) -> BatchPredictionResponse:  # noqa: B008 - FastAPI idiom
    predictor = _require_predictor()
    results = []
    for upload in files:
        image = await _read_image(upload)
        result = predictor.predict_image(image)
        results.append(PredictionResponse(filename=upload.filename, model_version=state["model_version"], **result))
    return BatchPredictionResponse(results=results)
