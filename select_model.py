"""
Model selection script for Pneumonia Classification.
Evaluates every checkpoint (.pth) in a models directory on a labelled evaluation set
(flat folder of images + labels CSV), ranks them by recall (PNEUMONIA = positive class),
logs each one to MLflow and registers the best one in the MLflow Model Registry
under the alias 'champion'.

Usage:
    python select_model.py
    python select_model.py --models_dir models --eval_dir data/test --labels_csv data/test/labels.csv
    python select_model.py --include_augmented --no_register
"""

import os
import sys
import glob
import argparse
from typing import Dict, Any, List, Optional, Tuple
import yaml
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data.dataset import get_transforms
from src.models.densenet import model_parameters as densenet_variants
from src.models.resnet import resnet_configurations as resnet_variants
from src.models.model_factory import get_model, EXTRA_ARCHITECTURES
from src.tracking.mlflow_tracker import Tracker

try:
    import mlflow
except ImportError:
    mlflow = None  # type: ignore[assignment]  # tracking is an optional dependency

REGISTERED_MODEL_NAME = "pneumonia-classifier"
CHAMPION_ALIAS = "champion"


class LabelledImageDataset(Dataset):
    """Flat folder of images + labels CSV (columns: filename, label_id)."""

    def __init__(self, image_dir: str, df: pd.DataFrame, transform):
        self.image_dir = image_dir
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(os.path.join(self.image_dir, row["filename"])).convert("RGB")
        return self.transform(image), int(row["label_id"])


def pick(cli_value, cfg_value, default):
    """CLI value wins, then config, then default. Unlike `or`, keeps 0 values."""
    if cli_value is not None:
        return cli_value
    return cfg_value if cfg_value is not None else default


def load_config(config_path: str) -> Dict[str, Any]:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank trained checkpoints on an evaluation set and pick a champion.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML configuration file.")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory containing .pth checkpoints.")
    parser.add_argument("--eval_dir", type=str, default="data/test", help="Folder with the evaluation images.")
    parser.add_argument("--labels_csv", type=str, default=None, help="Labels CSV (default: <eval_dir>/labels.csv).")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to save the ranking (default: results).")
    parser.add_argument("--batch_size", type=int, default=None, help="Evaluation batch size.")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader worker count.")
    parser.add_argument("--device", type=str, default=None, help="'cuda', 'cpu' or 'auto'.")
    parser.add_argument(
        "--include_augmented",
        action="store_true",
        help="Also evaluate on augmented copies (is_augmented == 1). Default: originals only.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N images (after the originals-only filter). "
             "For fast debugging/wiring checks - NOT for a real ranking decision.",
    )
    parser.add_argument("--no_register", action="store_true", help="Rank only; do not register a champion.")
    parser.add_argument("--no_mlflow", action="store_true", help="Disable MLflow tracking (implies --no_register).")
    return parser.parse_args(argv)


def infer_architecture(weights_path: str, num_classes: int, device) -> Tuple[str, torch.nn.Module]:
    """
    Finds which architecture a checkpoint belongs to by loading it strictly.
    The file name is used as a hint to try the most likely family first.
    """
    name = os.path.basename(weights_path).lower()
    dense = list(densenet_variants.keys())
    res = list(resnet_variants.keys())
    extra = list(EXTRA_ARCHITECTURES.keys())
    hinted = [a for a in dense + res if a in name]
    if "dense" in name:
        candidates = hinted + [a for a in dense if a not in hinted] + extra
    elif "res" in name:
        candidates = hinted + [a for a in res if a not in hinted]
    else:
        candidates = hinted + [a for a in dense + res if a not in hinted]

    errors = {}
    for arch in candidates:
        try:
            model = get_model(architecture=arch, num_classes=num_classes, weights_path=weights_path, device=device)
            return arch, model
        except Exception as e:  # size / key mismatch -> try next architecture
            errors[arch] = str(e).splitlines()[0][:80]
    raise RuntimeError(f"Could not match '{weights_path}' to any known architecture: {errors}")


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    probs: List[np.ndarray] = []
    for inputs, y in loader:
        out = model(inputs.to(device))
        probs.extend(torch.softmax(out, dim=1).cpu().numpy())
        preds.extend(out.argmax(1).cpu().numpy())
        labels.extend(y.numpy())
    return np.array(preds), np.array(labels), np.array(probs)


def compute_metrics(preds: np.ndarray, labels: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "recall": float(recall_score(labels, preds, pos_label=1, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "precision": float(precision_score(labels, preds, pos_label=1, zero_division=0)),
        "f1_score": float(f1_score(labels, preds, pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
        "roc_auc": float(roc_auc_score(labels, probs[:, 1])),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def main(argv=None) -> Dict[str, Any]:
    """
    Runs model selection. Callable from a pipeline: main([...cli args...]).
    Returns {'ranking': [...], 'champion': {...}, 'registered_version': str or None}.
    Raises FileNotFoundError if no checkpoints or evaluation data are found.
    """
    args = parse_args(argv)
    cfg = load_config(args.config) if args.config else {}
    data_cfg = cfg.get("data", {})
    eval_cfg = cfg.get("evaluation", {})
    num_classes = cfg.get("model", {}).get("num_classes", 2)

    output_dir = pick(args.output_dir, eval_cfg.get("results_dir"), "results")
    batch_size = pick(args.batch_size, eval_cfg.get("batch_size"), 32)
    num_workers = pick(args.num_workers, data_cfg.get("num_workers"), 4)
    device_arg = pick(args.device, eval_cfg.get("device"), "auto")
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)

    labels_csv = args.labels_csv or os.path.join(args.eval_dir, "labels.csv")
    if not os.path.isfile(labels_csv):
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")
    checkpoints = sorted(glob.glob(os.path.join(args.models_dir, "*.pth")))
    if not checkpoints:
        raise FileNotFoundError(f"No .pth checkpoints found in '{args.models_dir}'.")

    df = pd.read_csv(labels_csv)
    if not args.include_augmented and "is_augmented" in df.columns:
        df = df[df["is_augmented"] == 0]
    subset = "all images" if args.include_augmented else "originals only"
    if args.limit is not None:
        df = df.head(args.limit)
        subset += f", limited to {len(df)}"
        print(f"[WARN] --limit set: evaluating only {len(df)} images. For debugging only - "
              f"do not use this ranking to pick a production champion.")
    print("=" * 60)
    print(f"Model selection on {len(df)} images ({subset}) | device: {device}")
    print(f"Checkpoints: {len(checkpoints)} in '{args.models_dir}' | rank by: ROC-AUC")
    print("=" * 60)

    transform = get_transforms(
        data_cfg.get("image_size", 224),
        data_cfg.get("mean", [0.485, 0.456, 0.406]),
        data_cfg.get("std", [0.229, 0.224, 0.225]),
        split="val",
    )
    loader = DataLoader(
        LabelledImageDataset(args.eval_dir, df, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    tracker = Tracker.from_config(cfg, disabled=args.no_mlflow)
    results: List[Dict[str, Any]] = []
    for path in checkpoints:
        arch, model = infer_architecture(path, num_classes, device)
        print(f"\n>>> {os.path.basename(path)} (architecture: {arch})")
        preds, labels, probs = predict(model, loader, device)
        metrics = compute_metrics(preds, labels, probs)
        print(
            f"    recall={metrics['recall']:.4f}  specificity={metrics['specificity']:.4f}  "
            f"precision={metrics['precision']:.4f}  f1={metrics['f1_score']:.4f}  "
            f"acc={metrics['accuracy']:.4f}  auc={metrics['roc_auc']:.4f}"
        )

        with tracker.run(run_name=f"select-{os.path.splitext(os.path.basename(path))[0]}"):
            tracker.log_params({
                "model_architecture": arch,
                "weights_file": os.path.basename(path),
                "eval_set": os.path.basename(os.path.normpath(args.eval_dir)),
                "eval_subset": subset,
                "n_eval_images": len(df),
            })
            tracker.log_metrics({f"test_{k}": v for k, v in metrics.items()})
            run_id = tracker.run_id

        results.append({"weights": path, "architecture": arch, "run_id": run_id, **metrics})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Rank: ROC-AUC first (overall class separation), recall as tie-break
    ranking = sorted(results, key=lambda r: (r["roc_auc"], r["recall"]), reverse=True)
    os.makedirs(output_dir, exist_ok=True)
    ranking_df = pd.DataFrame(ranking).drop(columns=["run_id"])
    ranking_df.insert(0, "rank", range(1, len(ranking_df) + 1))
    csv_path = os.path.join(output_dir, "model_ranking.csv")
    ranking_df.to_csv(csv_path, index=False)

    print("\n================ Ranking (by ROC-AUC) ================")
    print(ranking_df[["rank", "weights", "architecture", "roc_auc", "recall", "specificity", "f1_score", "accuracy"]].to_string(index=False))
    champion = ranking[0]
    print(f"\n[CHAMPION] {champion['weights']} (roc_auc={champion['roc_auc']:.4f}, recall={champion['recall']:.4f}, specificity={champion['specificity']:.4f})")
    if champion["specificity"] < 0.5:
        print("[WARN] Champion specificity is below 0.5 - it may be flagging almost everything as PNEUMONIA.")
    print(f"[INFO] Ranking saved to: {csv_path}")

    # Register the champion in the MLflow Model Registry
    registered_version: Optional[str] = None
    if tracker.enabled and not args.no_register:
        registered_version = register_champion(tracker, champion, subset)
        print(f"[INFO] Registered '{REGISTERED_MODEL_NAME}' version {registered_version} with alias '{CHAMPION_ALIAS}'.")

    return {"ranking": ranking, "champion": champion, "registered_version": registered_version}


def register_champion(tracker: Tracker, champion: Dict[str, Any], subset: str) -> str:
    """Uploads the champion checkpoint to its run and registers it, alias = 'champion'."""
    mlflow.set_tracking_uri(tracker.tracking_uri)
    client = mlflow.MlflowClient()
    with mlflow.start_run(run_id=champion["run_id"]):
        mlflow.log_artifact(champion["weights"], artifact_path="model")
    try:
        client.create_registered_model(REGISTERED_MODEL_NAME)
    except mlflow.exceptions.MlflowException:
        pass  # already exists -> add a new version
    run = client.get_run(champion["run_id"])
    version = client.create_model_version(
        REGISTERED_MODEL_NAME, source=f"{run.info.artifact_uri}/model", run_id=champion["run_id"]
    )
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version.version, "architecture", champion["architecture"])
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version.version, "weights_file", os.path.basename(champion["weights"]))
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version.version, "selected_on", f"roc_auc, {subset}")
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, version.version)
    return str(version.version)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
