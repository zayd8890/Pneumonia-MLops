"""
Evaluation script for Pneumonia Classification.
Loads trained checkpoints and computes test accuracy, precision, recall, F1, ROC-AUC,
and outputs predictions CSV and metrics JSON.

Usage:
    python evaluate.py --model densenet201 --weights models/densenet201.pth --data_dir data
    python evaluate.py --model resnet18 --weights models/resnet18.pth --test_dir data/test
    python evaluate.py --config configs/config.yaml --weights models/densenet201.pth
"""

import os
import sys
import argparse
from typing import Dict, Any
import yaml
import torch

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.models.model_factory import get_model
from src.data.dataloader import get_dataloaders
from src.evaluation.evaluator import Evaluator
from src.tracking.mlflow_tracker import Tracker


def load_config(config_path: str) -> Dict[str, Any]:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def pick(cli_value, cfg_value, default):
    """CLI value wins, then config, then default. Unlike `or`, keeps 0 values."""
    if cli_value is not None:
        return cli_value
    return cfg_value if cfg_value is not None else default


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Pneumonia Detection Model (DenseNet / ResNet)")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model architecture ('densenet201', 'resnet18', etc.).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to pre-trained .pth model checkpoint.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Root directory containing 'test' split.",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default=None,
        help="Path directly to test images directory.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for evaluation (default: 32).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save evaluation results and predictions (default: 'results').",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="MLflow run id (returned by train.py) to log test metrics into. New run if omitted.",
    )
    parser.add_argument(
        "--no_mlflow",
        action="store_true",
        help="Disable MLflow experiment tracking.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="DataLoader worker count (default: from config).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('cuda', 'cpu', or 'auto').",
    )

    return parser.parse_args(argv)


def main(argv=None) -> Dict[str, Any]:
    """
    Runs evaluation. Callable from a pipeline: main([...cli args...]).
    Returns the metrics dict (accuracy, precision, recall, f1_score, roc_auc, ...).
    Raises FileNotFoundError if test data or weights are missing.
    """
    args = parse_args(argv)
    cfg = load_config(args.config) if args.config else {}

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    eval_cfg = cfg.get("evaluation", {})

    arch = pick(args.model, model_cfg.get("architecture"), "densenet201")
    num_classes = model_cfg.get("num_classes", 2)
    class_names = model_cfg.get("class_names", ["NORMAL", "PNEUMONIA"])

    data_dir = pick(args.data_dir, data_cfg.get("data_dir"), "data")
    test_dir = args.test_dir
    # An explicit --data_dir must not be overridden by test_dir from the config file
    if args.data_dir is None:
        test_dir = pick(test_dir, data_cfg.get("test_dir"), None)
    batch_size = pick(args.batch_size, eval_cfg.get("batch_size"), 32)
    output_dir = pick(args.output_dir, eval_cfg.get("results_dir"), "results")
    num_workers = pick(args.num_workers, data_cfg.get("num_workers"), 4)

    image_size = data_cfg.get("image_size", 224)
    mean = data_cfg.get("mean", [0.485, 0.456, 0.406])
    std = data_cfg.get("std", [0.229, 0.224, 0.225])

    device_arg = pick(args.device, eval_cfg.get("device"), "auto")
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)

    print("=" * 60)
    print(f"MLOps Evaluation: {arch.upper()}")
    print("=" * 60)
    print(f"Architecture:  {arch}")
    print(f"Weights:       {args.weights}")
    print(f"Data Dir:      {test_dir or data_dir}")
    print(f"Output Dir:    {output_dir}")
    print(f"Device:        {device}")
    print("=" * 60)

    # 1. Build test DataLoader
    _, _, test_loader, dataset_classes = get_dataloaders(
        data_dir=data_dir,
        test_dir=test_dir,
        batch_size=batch_size,
        image_size=image_size,
        mean=mean,
        std=std,
        num_workers=num_workers,
    )

    if test_loader is None:
        raise FileNotFoundError(
            f"Could not find test data in '{test_dir or data_dir}'. "
            "Place test images in '<data_dir>/test' (with class subfolders) or specify --test_dir."
        )

    # 2. Instantiate Model and Load Weights
    model = get_model(
        architecture=arch,
        num_classes=num_classes,
        weights_path=args.weights,
        device=device,
    )

    # 3. Run Evaluation
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=dataset_classes or class_names,
        results_dir=output_dir,
    )

    tracker = Tracker.from_config(cfg, disabled=args.no_mlflow)
    with tracker.run(run_name=f"{arch}-evaluate", run_id=args.run_id):
        metrics = evaluator.evaluate(model_name=arch)
        tracker.log_params({"eval_model_architecture": arch, "eval_weights": os.path.basename(args.weights)})
        tracker.log_metrics({
            f"test_{k}": metrics[k]
            for k in ("accuracy", "precision", "recall", "f1_score", "roc_auc")
            if k in metrics
        })
        tracker.log_artifact(os.path.join(output_dir, f"metrics_{arch}.json"), "evaluation")
        tracker.log_artifact(os.path.join(output_dir, f"test_predictions_{arch}.csv"), "evaluation")
        metrics["run_id"] = tracker.run_id
    print(f"[SUCCESS] Evaluation complete. Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
    return metrics


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
