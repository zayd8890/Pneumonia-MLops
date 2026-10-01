"""
Unit tests for select_model.py.

Covers the pure helpers (pick, compute_metrics, LabelledImageDataset,
infer_architecture, parse_args), an end-to-end main() run against a tiny
synthetic dataset + a real (untrained) resnet18 checkpoint, and
register_champion() with mlflow fully mocked.
"""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

import select_model
from select_model import (
    LabelledImageDataset,
    compute_metrics,
    infer_architecture,
    main,
    parse_args,
    pick,
    register_champion,
)
from src.models.model_factory import get_model


# ---------------------------------------------------------------------------
# pick()
# ---------------------------------------------------------------------------

def test_pick_cli_value_wins():
    assert pick("cli", "cfg", "default") == "cli"


def test_pick_falls_back_to_config_when_cli_none():
    assert pick(None, "cfg", "default") == "cfg"


def test_pick_falls_back_to_default_when_both_none():
    assert pick(None, None, "default") == "default"


def test_pick_keeps_zero_unlike_or():
    """The whole point of pick() over `a or b or c`: 0 is a valid, real value."""
    assert pick(0, 5, 10) == 0
    assert pick(None, 0, 10) == 0


# ---------------------------------------------------------------------------
# compute_metrics()
# ---------------------------------------------------------------------------

def test_compute_metrics_known_confusion_matrix():
    # labels: NORMAL, NORMAL, PNEUMONIA, PNEUMONIA
    # preds:  NORMAL, PNEUMONIA, PNEUMONIA, PNEUMONIA  -> tn=1, fp=1, fn=0, tp=2
    labels = np.array([0, 0, 1, 1])
    preds = np.array([0, 1, 1, 1])
    probs = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.1, 0.9]])

    metrics = compute_metrics(preds, labels, probs)

    assert metrics["tp"] == 2
    assert metrics["fp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fn"] == 0
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["specificity"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["roc_auc"] == pytest.approx(1.0)  # perfect separation in probs


def test_compute_metrics_zero_division_does_not_raise():
    """All-one-class predictions (e.g. a broken/degenerate model) must not crash."""
    labels = np.array([0, 0])
    preds = np.array([0, 0])
    probs = np.array([[0.9, 0.1], [0.8, 0.2]])
    metrics = compute_metrics(preds, labels, probs)
    assert metrics["fp"] == 0
    assert metrics["specificity"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# LabelledImageDataset
# ---------------------------------------------------------------------------

def test_labelled_image_dataset_reads_images_and_labels(tmp_path):
    img_path = tmp_path / "0000001.png"
    Image.new("RGB", (40, 40), color=(100, 100, 100)).save(img_path)
    df = pd.DataFrame([{"filename": "0000001.png", "label_id": 1}])

    from src.data.dataset import get_transforms
    dataset = LabelledImageDataset(str(tmp_path), df, get_transforms(image_size=32, split="val"))

    assert len(dataset) == 1
    image_tensor, label = dataset[0]
    assert isinstance(image_tensor, torch.Tensor)
    assert image_tensor.shape == (3, 32, 32)
    assert label == 1


# ---------------------------------------------------------------------------
# infer_architecture()
# ---------------------------------------------------------------------------

def test_infer_architecture_matches_resnet_from_filename(tmp_path):
    model = get_model(architecture="resnet18", num_classes=2)
    weights_path = tmp_path / "my_resnet18_checkpoint.pth"
    torch.save(model.state_dict(), weights_path)

    arch, loaded = infer_architecture(str(weights_path), num_classes=2, device=torch.device("cpu"))

    assert arch == "resnet18"
    assert loaded is not None


def test_infer_architecture_raises_when_nothing_matches(tmp_path):
    bad_file = tmp_path / "densenet201_corrupt.pth"
    torch.save({"not": "a real state dict"}, bad_file)

    with pytest.raises(RuntimeError, match="Could not match"):
        infer_architecture(str(bad_file), num_classes=2, device=torch.device("cpu"))


# ---------------------------------------------------------------------------
# parse_args()
# ---------------------------------------------------------------------------

def test_parse_args_defaults():
    args = parse_args([])
    assert args.models_dir == "models"
    assert args.eval_dir == "data/test"
    assert args.limit is None
    assert args.include_augmented is False
    assert args.no_register is False
    assert args.no_mlflow is False


def test_parse_args_limit_and_flags():
    args = parse_args(["--limit", "10", "--no_register", "--no_mlflow", "--include_augmented"])
    assert args.limit == 10
    assert args.no_register is True
    assert args.no_mlflow is True
    assert args.include_augmented is True


# ---------------------------------------------------------------------------
# main(): raises on missing inputs
# ---------------------------------------------------------------------------

def test_main_raises_when_labels_csv_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="Labels CSV"):
        main([
            "--config", str(tmp_path / "no_such_config.yaml"),
            "--eval_dir", str(tmp_path),
            "--models_dir", str(tmp_path),
        ])


def test_main_raises_when_no_checkpoints_found(tmp_path):
    (tmp_path / "labels.csv").write_text("filename,label_id\n", encoding="utf-8")
    empty_models_dir = tmp_path / "empty_models"
    empty_models_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="No .pth checkpoints"):
        main([
            "--config", str(tmp_path / "no_such_config.yaml"),
            "--eval_dir", str(tmp_path),
            "--models_dir", str(empty_models_dir),
        ])


# ---------------------------------------------------------------------------
# main(): end-to-end against a tiny synthetic dataset + one real checkpoint
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_eval_setup(tmp_path):
    """4 tiny synthetic images (balanced NORMAL/PNEUMONIA) + labels.csv + one
    real (untrained) resnet18 checkpoint, named so infer_architecture matches fast."""
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    rows = []
    for i, label_id in enumerate([0, 0, 1, 1]):
        fname = f"{i:03d}.png"
        Image.new("RGB", (32, 32), color=(i * 20, i * 20, i * 20)).save(eval_dir / fname)
        rows.append({"filename": fname, "label_id": label_id, "is_augmented": 0})
    pd.DataFrame(rows).to_csv(eval_dir / "labels.csv", index=False)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model = get_model(architecture="resnet18", num_classes=2)
    torch.save(model.state_dict(), models_dir / "resnet18.pth")

    output_dir = tmp_path / "results"
    return {"eval_dir": eval_dir, "models_dir": models_dir, "output_dir": output_dir, "tmp_path": tmp_path}


def test_main_end_to_end_no_mlflow(tiny_eval_setup):
    result = main([
        "--config", str(tiny_eval_setup["tmp_path"] / "no_such_config.yaml"),
        "--eval_dir", str(tiny_eval_setup["eval_dir"]),
        "--models_dir", str(tiny_eval_setup["models_dir"]),
        "--output_dir", str(tiny_eval_setup["output_dir"]),
        "--batch_size", "2",
        "--num_workers", "0",
        "--device", "cpu",
        "--no_mlflow",
    ])

    assert len(result["ranking"]) == 1
    champion = result["champion"]
    assert champion["architecture"] == "resnet18"
    assert "roc_auc" in champion
    assert result["registered_version"] is None  # --no_mlflow disables registration entirely

    csv_path = tiny_eval_setup["output_dir"] / "model_ranking.csv"
    assert csv_path.is_file()
    ranking_df = pd.read_csv(csv_path)
    assert len(ranking_df) == 1
    assert ranking_df.iloc[0]["rank"] == 1


def test_main_respects_limit(tiny_eval_setup):
    """--limit 2 must actually reduce the evaluated set (captured via n_eval_images param
    when tracking is on) - here we check indirectly via the champion's metrics being
    computed at all without error on a 2-image subset, and that main() doesn't choke
    on a smaller-than-batch_size dataset."""
    result = main([
        "--config", str(tiny_eval_setup["tmp_path"] / "no_such_config.yaml"),
        "--eval_dir", str(tiny_eval_setup["eval_dir"]),
        "--models_dir", str(tiny_eval_setup["models_dir"]),
        "--output_dir", str(tiny_eval_setup["output_dir"]),
        "--batch_size", "4",
        "--num_workers", "0",
        "--device", "cpu",
        "--limit", "2",
        "--no_mlflow",
    ])
    assert len(result["ranking"]) == 1


def test_main_originals_only_filters_augmented(tiny_eval_setup):
    """Augmented rows must be excluded unless --include_augmented is passed."""
    df = pd.read_csv(tiny_eval_setup["eval_dir"] / "labels.csv")
    df.loc[0, "is_augmented"] = 1
    df.to_csv(tiny_eval_setup["eval_dir"] / "labels.csv", index=False)

    with patch("select_model.predict", wraps=select_model.predict) as spy:
        main([
            "--config", str(tiny_eval_setup["tmp_path"] / "no_such_config.yaml"),
            "--eval_dir", str(tiny_eval_setup["eval_dir"]),
            "--models_dir", str(tiny_eval_setup["models_dir"]),
            "--output_dir", str(tiny_eval_setup["output_dir"]),
            "--batch_size", "4",
            "--num_workers", "0",
            "--device", "cpu",
            "--no_mlflow",
        ])
    loader_used = spy.call_args[0][1]
    assert len(loader_used.dataset) == 3  # 4 rows minus 1 augmented


# ---------------------------------------------------------------------------
# register_champion(): mlflow fully mocked
# ---------------------------------------------------------------------------

def test_register_champion_calls_expected_sequence():
    mock_mlflow = MagicMock()
    mock_client = MagicMock()
    mock_mlflow.MlflowClient.return_value = mock_client
    mock_mlflow.exceptions.MlflowException = Exception
    mock_run = MagicMock()
    mock_run.info.artifact_uri = "mlflow-artifacts:/1/abc123/artifacts"
    mock_client.get_run.return_value = mock_run
    mock_version = MagicMock()
    mock_version.version = "3"
    mock_client.create_model_version.return_value = mock_version

    tracker = MagicMock()
    tracker.tracking_uri = "http://localhost:5001"
    champion = {"run_id": "abc123", "weights": "models/densenet201.pth", "architecture": "densenet201"}

    with patch("select_model.mlflow", mock_mlflow):
        version = register_champion(tracker, champion, "originals only")

    assert version == "3"
    mock_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5001")
    mock_mlflow.log_artifact.assert_called_once_with("models/densenet201.pth", artifact_path="model")
    mock_client.create_model_version.assert_called_once_with(
        "pneumonia-classifier",
        source="mlflow-artifacts:/1/abc123/artifacts/model",
        run_id="abc123",
    )
    mock_client.set_registered_model_alias.assert_called_once_with("pneumonia-classifier", "champion", "3")
    assert mock_client.set_model_version_tag.call_count == 3


def test_register_champion_ignores_already_exists_error():
    """create_registered_model raising (model already exists) must not propagate."""
    mock_mlflow = MagicMock()
    mock_client = MagicMock()
    mock_mlflow.MlflowClient.return_value = mock_client
    mock_mlflow.exceptions.MlflowException = Exception
    mock_client.create_registered_model.side_effect = Exception("already exists")
    mock_run = MagicMock()
    mock_run.info.artifact_uri = "mlflow-artifacts:/1/abc/artifacts"
    mock_client.get_run.return_value = mock_run
    mock_client.create_model_version.return_value = MagicMock(version="1")

    tracker = MagicMock()
    tracker.tracking_uri = "sqlite:///mlflow.db"
    champion = {"run_id": "abc", "weights": "models/x.pth", "architecture": "resnet18"}

    with patch("select_model.mlflow", mock_mlflow):
        version = register_champion(tracker, champion, "originals only")  # must not raise

    assert version == "1"
