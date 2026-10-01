"""
Unit tests for the Tracker class (src/tracking/mlflow_tracker.py).

Uses a mix of:
- mocked mlflow, to verify the Tracker never touches mlflow when disabled
- a real local sqlite tracking URI, to verify params/metrics/artifacts actually
  land correctly when enabled (fast and network-free with a sqlite backend)
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.tracking.mlflow_tracker import Tracker


# ---------------------------------------------------------------------------
# enabled / disabled resolution
# ---------------------------------------------------------------------------

def test_disabled_via_constructor_flag():
    tracker = Tracker(enabled=False)
    assert tracker.enabled is False


def test_enabled_when_mlflow_is_installed():
    tracker = Tracker(enabled=True)
    assert tracker.enabled is True


def test_disabled_when_mlflow_not_installed():
    with patch("src.tracking.mlflow_tracker.mlflow", None):
        tracker = Tracker(enabled=True)
        assert tracker.enabled is False


def test_from_config_reads_mlflow_section():
    cfg = {
        "mlflow": {
            "enabled": True,
            "tracking_uri": "sqlite:///custom.db",
            "experiment_name": "my-experiment",
        }
    }
    tracker = Tracker.from_config(cfg)
    assert tracker.tracking_uri == "sqlite:///custom.db"
    assert tracker.experiment_name == "my-experiment"


def test_from_config_defaults_when_no_mlflow_section():
    tracker = Tracker.from_config({})
    assert tracker.tracking_uri == "sqlite:///mlflow.db"
    assert tracker.experiment_name == "pneumonia-classification"


def test_from_config_disabled_flag_overrides_config():
    cfg = {"mlflow": {"enabled": True}}
    tracker = Tracker.from_config(cfg, disabled=True)
    assert tracker.enabled is False


def test_from_config_env_var_overrides_config_file():
    """MLFLOW_TRACKING_URI must win over configs/config.yaml's tracking_uri -
    this is the exact bug that caused select_model.py to silently register
    against the wrong (local sqlite) database instead of the Docker/Postgres server."""
    cfg = {"mlflow": {"tracking_uri": "sqlite:///from_config.db"}}
    with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://localhost:5001"}):
        tracker = Tracker.from_config(cfg)
    assert tracker.tracking_uri == "http://localhost:5001"


def test_from_config_no_env_var_falls_back_to_config():
    cfg = {"mlflow": {"tracking_uri": "sqlite:///from_config.db"}}
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("MLFLOW_TRACKING_URI", None)
        tracker = Tracker.from_config(cfg)
    assert tracker.tracking_uri == "sqlite:///from_config.db"


# ---------------------------------------------------------------------------
# disabled tracker: every method must be a true no-op (never touch mlflow)
# ---------------------------------------------------------------------------

def test_disabled_tracker_never_calls_mlflow():
    mock_mlflow = MagicMock()
    with patch("src.tracking.mlflow_tracker.mlflow", mock_mlflow):
        tracker = Tracker(enabled=False)
        with tracker.run(run_name="should-not-run"):
            tracker.log_params({"lr": 0.001})
            tracker.log_metrics({"accuracy": 0.9})
            tracker.log_artifact(__file__)  # a real file, to isolate the enabled check
    mock_mlflow.start_run.assert_not_called()
    mock_mlflow.log_params.assert_not_called()
    mock_mlflow.log_metrics.assert_not_called()
    mock_mlflow.log_artifact.assert_not_called()


def test_disabled_tracker_run_still_yields_self():
    tracker = Tracker(enabled=False)
    with tracker.run() as t:
        assert t is tracker
    assert tracker.run_id is None


# ---------------------------------------------------------------------------
# enabled tracker, mocked mlflow: verify the right calls happen with the
# right argument transforms
# ---------------------------------------------------------------------------

def test_log_params_converts_none_to_string():
    """mlflow.log_params rejects None values outright - the wrapper must stringify them."""
    mock_mlflow = MagicMock()
    with patch("src.tracking.mlflow_tracker.mlflow", mock_mlflow):
        tracker = Tracker(enabled=True)
        tracker.log_params({"weights_path": None, "architecture": "densenet201"})
    mock_mlflow.log_params.assert_called_once_with(
        {"weights_path": "None", "architecture": "densenet201"}
    )


def test_log_metrics_casts_to_float():
    mock_mlflow = MagicMock()
    with patch("src.tracking.mlflow_tracker.mlflow", mock_mlflow):
        tracker = Tracker(enabled=True)
        tracker.log_metrics({"accuracy": 1, "loss": "0.5"}, step=3)
    args, kwargs = mock_mlflow.log_metrics.call_args
    assert args[0] == {"accuracy": 1.0, "loss": 0.5}
    assert all(isinstance(v, float) for v in args[0].values())
    assert kwargs["step"] == 3


def test_log_artifact_skips_missing_file():
    mock_mlflow = MagicMock()
    with patch("src.tracking.mlflow_tracker.mlflow", mock_mlflow):
        tracker = Tracker(enabled=True)
        tracker.log_artifact("/no/such/file/on/disk.pth")
    mock_mlflow.log_artifact.assert_not_called()


def test_log_artifact_uploads_existing_file():
    mock_mlflow = MagicMock()
    with patch("src.tracking.mlflow_tracker.mlflow", mock_mlflow):
        tracker = Tracker(enabled=True)
        tracker.log_artifact(__file__, artifact_path="code")
    mock_mlflow.log_artifact.assert_called_once_with(__file__, artifact_path="code")


def test_run_resumes_existing_run_id_without_a_new_run_name():
    """Passing run_id should resume that run, not create a new named one -
    this is what register_champion() in select_model.py relies on."""
    mock_mlflow = MagicMock()
    mock_active = MagicMock()
    mock_active.info.run_id = "abc123"
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_active
    with patch("src.tracking.mlflow_tracker.mlflow", mock_mlflow):
        tracker = Tracker(enabled=True)
        with tracker.run(run_name="ignored-when-resuming", run_id="abc123"):
            pass
    mock_mlflow.start_run.assert_called_once_with(run_id="abc123", run_name=None)
    assert tracker.run_id == "abc123"


# ---------------------------------------------------------------------------
# real end-to-end check against a local sqlite tracking URI (fast, no network)
# ---------------------------------------------------------------------------

def test_real_sqlite_run_logs_params_and_metrics(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    db_path = tmp_path / "test_tracking.db"
    tracker = Tracker(
        enabled=True,
        tracking_uri=f"sqlite:///{db_path}",
        experiment_name="pytest-experiment",
    )
    with tracker.run(run_name="pytest-run"):
        tracker.log_params({"architecture": "resnet18", "epochs": 1})
        tracker.log_metrics({"accuracy": 0.5})
        run_id = tracker.run_id

    client = mlflow.MlflowClient(tracking_uri=f"sqlite:///{db_path}")
    run = client.get_run(run_id)
    assert run.data.params["architecture"] == "resnet18"
    assert run.data.metrics["accuracy"] == 0.5
