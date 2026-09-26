"""
Training script for Pneumonia Classification.
Supports both DenseNet and ResNet architectures with CLI arguments and YAML configuration.

Usage:
    python train.py --model densenet201 --data_dir data --epochs 20 --batch_size 32
    python train.py --model resnet18 --data_dir data --epochs 20 --lr 0.001
    python train.py --config configs/config.yaml
"""

import os
import sys
import argparse
from typing import Dict, Any
import yaml
import torch
import torch.nn as nn

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.models.model_factory import get_model
from src.data.dataloader import get_dataloaders
from src.training.trainer import Trainer
from src.tracking.mlflow_tracker import Tracker


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file if it exists."""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def pick(cli_value, cfg_value, default):
    """CLI value wins, then config, then default. Unlike `or`, keeps 0 / 0.0 values."""
    if cli_value is not None:
        return cli_value
    return cfg_value if cfg_value is not None else default


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Pneumonia Detection Model (DenseNet / ResNet)")
    
    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file.",
    )

    # Architecture selection
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model architecture ('densenet201', 'densenet121', 'resnet18', etc.).",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Number of classification categories (default: 2).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional path to existing .pth weights to resume/finetune.",
    )

    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Root directory containing train/val splits.",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default=None,
        help="Path to training images directory.",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default=None,
        help="Path to validation images directory.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="Image input resolution (default: 224).",
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (default: 20).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (default: 32).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate for Adam optimizer (default: 0.001).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="DataLoader worker count (default: 4).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('cuda', 'cpu', or 'auto').",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save checkpoints (default: 'models').",
    )

    parser.add_argument(
        "--no_mlflow",
        action="store_true",
        help="Disable MLflow experiment tracking.",
    )

    return parser.parse_args(argv)


def main(argv=None) -> Dict[str, Any]:
    """
    Runs training. Callable from a pipeline: main([...cli args...]).
    Returns {'run_id' (MLflow, or None), 'model_path', 'best_model_path', 'history_path', 'history', 'architecture', 'class_names'}.
    Raises FileNotFoundError if no training data is found.
    """
    args = parse_args(argv)
    cfg = load_config(args.config) if args.config else {}

    # Extract configuration with CLI overrides
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})

    arch = pick(args.model, model_cfg.get("architecture"), "densenet201")
    num_classes = pick(args.num_classes, model_cfg.get("num_classes"), 2)
    weights_path = pick(args.weights, model_cfg.get("weights_path"), None)

    data_dir = pick(args.data_dir, data_cfg.get("data_dir"), "data")
    train_dir = args.train_dir
    val_dir = args.val_dir
    # An explicit --data_dir must not be overridden by train/val dirs from the config file
    if args.data_dir is None:
        train_dir = pick(train_dir, data_cfg.get("train_dir"), None)
        val_dir = pick(val_dir, data_cfg.get("val_dir"), None)
    image_size = pick(args.image_size, data_cfg.get("image_size"), 224)
    batch_size = pick(args.batch_size, data_cfg.get("batch_size"), 32)
    num_workers = pick(args.num_workers, data_cfg.get("num_workers"), 4)
    mean = data_cfg.get("mean", [0.485, 0.456, 0.406])
    std = data_cfg.get("std", [0.229, 0.224, 0.225])

    epochs = pick(args.epochs, train_cfg.get("epochs"), 20)
    lr = pick(args.lr, train_cfg.get("learning_rate"), 0.001)
    save_dir = pick(args.save_dir, train_cfg.get("save_dir"), "models")
    device_arg = pick(args.device, train_cfg.get("device"), "auto")

    # Resolve device
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)

    print("=" * 60)
    print(f"MLOps Training Pipeline: {arch.upper()}")
    print("=" * 60)
    print(f"Architecture:      {arch}")
    print(f"Classes:           {num_classes}")
    print(f"Epochs:            {epochs}")
    print(f"Batch Size:        {batch_size}")
    print(f"Learning Rate:     {lr}")
    print(f"Device:            {device}")
    print(f"Dataset Location:  {data_dir}")
    print(f"Save Directory:    {save_dir}")
    print("=" * 60)

    # 1. Build DataLoaders
    train_loader, val_loader, _, class_names = get_dataloaders(
        data_dir=data_dir,
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=batch_size,
        image_size=image_size,
        mean=mean,
        std=std,
        num_workers=num_workers,
    )

    if train_loader is None:
        raise FileNotFoundError(
            f"Could not find training data in '{train_dir or data_dir}'. "
            "Place the dataset in '<data_dir>/train' (with class subfolders) or specify --data_dir."
        )

    print(f"[INFO] Classes detected: {class_names}")
    print(f"[INFO] Training batches: {len(train_loader)}")
    if val_loader:
        print(f"[INFO] Validation batches: {len(val_loader)}")

    # 2. Build Model
    model = get_model(
        architecture=arch,
        num_classes=num_classes,
        weights_path=weights_path,
        device=device,
    )

    # 3. Setup Loss and Optimizer matching notebooks
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 4. Train Model
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        save_dir=save_dir,
        model_name=arch,
    )

    tracker = Tracker.from_config(cfg, disabled=args.no_mlflow)
    with tracker.run(run_name=f"{arch}-train"):
        tracker.log_params({
            "model_architecture": arch,
            "learning_rate": lr,
            "batch_size": batch_size,
            "optimizer": type(optimizer).__name__.lower(),
            "epochs": epochs,
            "num_classes": num_classes,
            "image_size": image_size,
            "loss_function": type(criterion).__name__,
            "weights_path": weights_path,
        })

        history = trainer.fit()

        # Per-epoch loss / accuracy curves + final values
        for step in range(len(history["train_loss"])):
            epoch_metrics = {"train_loss": history["train_loss"][step], "train_acc": history["train_acc"][step]}
            if step < len(history["val_loss"]):
                epoch_metrics["val_loss"] = history["val_loss"][step]
                epoch_metrics["val_acc"] = history["val_acc"][step]
            tracker.log_metrics(epoch_metrics, step=step + 1)
        final = {"final_train_loss": history["train_loss"][-1], "final_train_acc": history["train_acc"][-1]}
        if history["val_loss"]:
            final.update({"final_val_loss": history["val_loss"][-1], "final_val_acc": history["val_acc"][-1]})
        tracker.log_metrics(final)

        latest_path = os.path.join(save_dir, f"{arch}.pth")
        best_path = os.path.join(save_dir, f"{arch}_best.pth")
        tracker.log_artifact(latest_path, "model")
        tracker.log_artifact(best_path, "model")
        tracker.log_artifact(os.path.join(save_dir, f"{arch}_history.json"), "model")
        run_id = tracker.run_id

    print("[SUCCESS] Training pipeline finished successfully.")
    return {
        "run_id": run_id,
        "architecture": arch,
        "class_names": class_names,
        "model_path": latest_path,
        "best_model_path": best_path if os.path.isfile(best_path) else latest_path,
        "history_path": os.path.join(save_dir, f"{arch}_history.json"),
        "history": history,
    }


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
