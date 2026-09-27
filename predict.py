"""
Inference CLI script for Pneumonia Classification.
Supports both single image and batch directory inference using DenseNet or ResNet.

Usage:
    # Single image prediction
    python predict.py --model densenet201 --weights models/densenet201.pth --image sample.jpeg

    # Batch prediction on a folder of images
    python predict.py --model resnet18 --weights models/resnet18.pth --image_dir data/test/PNEUMONIA --output results/predictions.csv
"""

import os
import sys
import glob
import argparse
from typing import Dict, Any, List
import yaml

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.inference.predictor import Predictor


def load_config(config_path: str) -> Dict[str, Any]:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pneumonia Inference CLI (DenseNet / ResNet)")

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
        help="Path to pre-trained .pth model weights file.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to single image file for inference.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Directory containing images to process in batch.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save prediction results (CSV or JSON).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cuda', 'cpu', or 'auto').",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config) if args.config else {}

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    inf_cfg = cfg.get("inference", {})

    arch = args.model or model_cfg.get("architecture", "densenet201")
    num_classes = model_cfg.get("num_classes", 2)
    class_names = model_cfg.get("class_names", ["NORMAL", "PNEUMONIA"])
    image_size = data_cfg.get("image_size", 224)
    mean = data_cfg.get("mean", [0.485, 0.456, 0.406])
    std = data_cfg.get("std", [0.229, 0.224, 0.225])

    device_arg = args.device or inf_cfg.get("device", "auto")

    if not args.image and not args.image_dir:
        print("[ERROR] Please provide either --image or --image_dir for inference.")
        return

    # Initialize Predictor
    predictor = Predictor(
        architecture=arch,
        weights_path=args.weights,
        num_classes=num_classes,
        class_names=class_names,
        image_size=image_size,
        mean=mean,
        std=std,
        device=device_arg,
    )

    # 1. Single Image Inference
    if args.image:
        if not os.path.exists(args.image):
            print(f"[ERROR] Image not found: {args.image}")
            return

        result = predictor.predict_image(args.image)
        print("\n================ Prediction Result ================")
        print(f"Image:            {args.image}")
        print(f"Predicted Class:  {result['predicted_label']} (ID: {result['predicted_class_id']})")
        print(f"Confidence:       {result['confidence'] * 100:.2f}%")
        print("Probabilities:")
        for cname, prob in result["probabilities"].items():
            print(f"  - {cname}: {prob * 100:.2f}%")
        print("===================================================")

    # 2. Batch Inference
    elif args.image_dir:
        if not os.path.exists(args.image_dir):
            print(f"[ERROR] Directory not found: {args.image_dir}")
            return

        extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")
        image_paths: List[str] = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(args.image_dir, ext)))
            image_paths.extend(glob.glob(os.path.join(args.image_dir, "**", ext), recursive=True))

        image_paths = sorted(list(set(image_paths)))
        if not image_paths:
            print(f"[ERROR] No valid image files found in '{args.image_dir}'.")
            return

        print(f"[INFO] Found {len(image_paths)} images. Running batch inference...")
        output_csv = args.output or os.path.join("results", f"batch_predictions_{arch}.csv")
        df = predictor.predict_batch(image_paths, output_csv_path=output_csv)
        print(f"[SUCCESS] Batch predictions completed for {len(df)} images.")
        print(df.head())


if __name__ == "__main__":
    main()
