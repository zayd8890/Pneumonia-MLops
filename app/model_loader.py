"""
Loads the serving model.

Preferred path: pull the 'champion'-aliased version from the MLflow Model Registry
(the model select_model.py registers). Falls back to a local .pth file (MODEL_WEIGHTS_PATH
/ MODEL_ARCHITECTURE env vars, or configs/config.yaml) when no registry is reachable -
this keeps the API runnable in CI or before any model has been registered.
"""

import os
from typing import Optional, Tuple

import yaml

from src.inference.predictor import Predictor

REGISTERED_MODEL_NAME = "pneumonia-classifier"
CHAMPION_ALIAS = "champion"


def load_config(config_path: str = "configs/config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _try_load_from_registry(cfg: dict) -> Tuple[Optional[Predictor], Optional[str]]:
    """Returns (predictor, model_version) if the registry has a 'champion', else (None, None)."""
    mcfg = cfg.get("mlflow", {}) or {}
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", mcfg.get("tracking_uri", "sqlite:///mlflow.db"))
    try:
        import mlflow
        import torch

        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.MlflowClient()
        mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
        arch = client.get_model_version(REGISTERED_MODEL_NAME, mv.version).tags.get("architecture", "densenet201")

        local_path = mlflow.artifacts.download_artifacts(f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}")
        # The registered artifact is a directory; find the .pth file inside it.
        weights_file = None
        for root, _dirs, files in os.walk(local_path):
            for f in files:
                if f.endswith(".pth"):
                    weights_file = os.path.join(root, f)
                    break
        if weights_file is None:
            raise FileNotFoundError(f"No .pth file found under registry artifact path: {local_path}")

        data_cfg = cfg.get("data", {})
        model_cfg = cfg.get("model", {})
        predictor = Predictor(
            architecture=arch,
            weights_path=weights_file,
            num_classes=model_cfg.get("num_classes", 2),
            class_names=model_cfg.get("class_names", ["NORMAL", "PNEUMONIA"]),
            image_size=data_cfg.get("image_size", 224),
            mean=data_cfg.get("mean"),
            std=data_cfg.get("std"),
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        return predictor, f"{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS} (v{mv.version}, {arch})"
    except Exception as e:  # registry unreachable, not configured, or nothing registered yet
        print(f"[WARN] Could not load champion from MLflow Registry: {e}")
        return None, None


def _load_from_local_weights(cfg: dict) -> Tuple[Optional[Predictor], Optional[str]]:
    """Fallback: a fixed .pth file, via env vars or configs/config.yaml."""
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    weights_path = os.environ.get("MODEL_WEIGHTS_PATH", model_cfg.get("weights_path"))
    architecture = os.environ.get("MODEL_ARCHITECTURE", model_cfg.get("architecture", "densenet201"))

    if not weights_path or not os.path.isfile(weights_path):
        return None, None

    import torch

    predictor = Predictor(
        architecture=architecture,
        weights_path=weights_path,
        num_classes=model_cfg.get("num_classes", 2),
        class_names=model_cfg.get("class_names", ["NORMAL", "PNEUMONIA"]),
        image_size=data_cfg.get("image_size", 224),
        mean=data_cfg.get("mean"),
        std=data_cfg.get("std"),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    return predictor, f"local:{os.path.basename(weights_path)} ({architecture})"


def load_serving_model(config_path: str = "configs/config.yaml") -> Tuple[Optional[Predictor], Optional[str]]:
    """
    Returns (predictor, model_version_label). Tries the MLflow registry champion first,
    then a local weights file. Returns (None, None) if neither is available - the API
    starts up but /predict returns 503 until a model exists.
    """
    cfg = load_config(config_path)
    predictor, version = _try_load_from_registry(cfg)
    if predictor is not None:
        return predictor, version
    return _load_from_local_weights(cfg)
