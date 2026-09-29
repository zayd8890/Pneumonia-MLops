"""
Thin MLflow wrapper used by train.py and evaluate.py.

Tracking is optional: if it is disabled (config / --no_mlflow) or MLflow is not
installed, every method becomes a no-op so training and evaluation still run.

A single MLflow run holds one experiment: the hyperparameters (learning rate, batch size,
optimizer, epochs, architecture), the metrics (accuracy, precision, recall, F1, loss) and the
model artifact. train.py creates the run and returns its id; evaluate.py can join the same
run with --run_id so train and test results end up in one place.
"""

import os
from contextlib import contextmanager
from typing import Any, Dict, Optional

try:
    import mlflow
except ImportError:  # tracking is optional
    mlflow = None


class Tracker:
    def __init__(
        self,
        enabled: bool = True,
        tracking_uri: str = "sqlite:///mlflow.db",
        experiment_name: str = "pneumonia-classification",
    ):
        self.enabled = bool(enabled) and mlflow is not None
        if enabled and mlflow is None:
            print("[WARN] mlflow is not installed - experiment tracking disabled.")
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_id: Optional[str] = None

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], disabled: bool = False) -> "Tracker":
        mcfg = cfg.get("mlflow", {}) or {}
        return cls(
            enabled=mcfg.get("enabled", True) and not disabled,
            # MLFLOW_TRACKING_URI env var wins if set (e.g. pointing scripts at a
            # Docker/Postgres-backed server), else fall back to the config file.
            tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", mcfg.get("tracking_uri", "sqlite:///mlflow.db")),
            experiment_name=mcfg.get("experiment_name", "pneumonia-classification"),
        )

    @contextmanager
    def run(self, run_name: Optional[str] = None, run_id: Optional[str] = None):
        """Starts a new run, or resumes `run_id`. Yields self; self.run_id is set."""
        if not self.enabled:
            yield self
            return
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_id=run_id, run_name=None if run_id else run_name) as active:
            self.run_id = active.info.run_id
            yield self

    def log_params(self, params: Dict[str, Any]) -> None:
        if self.enabled:
            mlflow.log_params({k: ("None" if v is None else v) for k, v in params.items()})

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self.enabled:
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def log_artifact(self, path: str, artifact_path: Optional[str] = None) -> None:
        if self.enabled and os.path.isfile(path):
            mlflow.log_artifact(path, artifact_path=artifact_path)
