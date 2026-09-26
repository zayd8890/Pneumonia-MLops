"""
Inference module for running predictions on individual images or batches.
"""

import os
from typing import Union, List, Dict, Any, Optional
import torch
import torch.nn as nn
from PIL import Image
import pandas as pd

from src.data.dataset import get_transforms
from src.models.model_factory import get_model


class Predictor:
    """
    Predictor class for image-based pneumonia inference.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        architecture: str = "densenet201",
        weights_path: Optional[str] = None,
        num_classes: int = 2,
        class_names: Optional[List[str]] = None,
        image_size: int = 224,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        device: Optional[Union[str, torch.device]] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(
                "cuda" if (device == "auto" and torch.cuda.is_available()) or device == "cuda" else "cpu"
            )
        else:
            self.device = device

        self.class_names = class_names or ["NORMAL", "PNEUMONIA"]
        self.transform = get_transforms(image_size=image_size, mean=mean, std=std, split="val")

        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = get_model(
                architecture=architecture,
                num_classes=num_classes,
                weights_path=weights_path,
                device=self.device,
            )

        self.model.eval()

    def predict_image(self, image_input: Union[str, Image.Image]) -> Dict[str, Any]:
        """
        Run inference on a single image.

        Args:
            image_input: File path to image or PIL.Image instance.

        Returns:
            Dictionary with prediction details:
            {
                'predicted_class_id': int,
                'predicted_label': str,
                'confidence': float,
                'probabilities': dict of {class_name: float}
            }
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found at: {image_input}")
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1).squeeze(0).cpu().numpy()
            pred_id = int(np_argmax(probs))

        pred_label = self.class_names[pred_id] if pred_id < len(self.class_names) else str(pred_id)
        confidence = float(probs[pred_id])

        prob_dict = {
            self.class_names[i] if i < len(self.class_names) else f"Class_{i}": float(probs[i])
            for i in range(len(probs))
        }

        return {
            "predicted_class_id": pred_id,
            "predicted_label": pred_label,
            "confidence": confidence,
            "probabilities": prob_dict,
        }

    def predict_batch(
        self,
        image_paths: List[str],
        output_csv_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Run inference on a list of image paths and return a DataFrame.

        Args:
            image_paths: List of file paths.
            output_csv_path: Optional path to save CSV.

        Returns:
            DataFrame with predictions and confidence scores.
        """
        records = []
        for path in image_paths:
            try:
                res = self.predict_image(path)
                rec = {
                    "image_path": path,
                    "predicted_class_id": res["predicted_class_id"],
                    "predicted_label": res["predicted_label"],
                    "confidence": res["confidence"],
                }
                for class_name, prob in res["probabilities"].items():
                    rec[f"prob_{class_name}"] = prob
                records.append(rec)
            except Exception as e:
                records.append({
                    "image_path": path,
                    "error": str(e),
                })

        df = pd.DataFrame(records)
        if output_csv_path:
            os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
            df.to_csv(output_csv_path, index=False)
            print(f"[INFO] Batch predictions saved to: {output_csv_path}")

        return df


def np_argmax(arr: Any) -> int:
    """Lightweight argmax helper."""
    max_idx = 0
    max_val = arr[0]
    for i in range(1, len(arr)):
        if arr[i] > max_val:
            max_val = arr[i]
            max_idx = i
    return max_idx
