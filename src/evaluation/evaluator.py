"""
Evaluation module for computing metrics and saving predictions.
Preserves DataFrame prediction schema from the notebooks.
"""

import os
import json
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)


class Evaluator:
    """
    Evaluator that matches the notebooks' testing and prediction loops,
    exporting predictions DataFrame and comprehensive MLOps metrics.
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: Optional[torch.device] = None,
        class_names: Optional[List[str]] = None,
        results_dir: str = "results",
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.test_loader = test_loader
        self.class_names = class_names or ["NORMAL", "PNEUMONIA"]
        self.results_dir = results_dir

        os.makedirs(self.results_dir, exist_ok=True)

    def get_predictions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs evaluation on test_loader and extracts predictions, true labels, and probabilities.
        Matches notebook implementation exactly.
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                probs = torch.softmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        return np.array(all_preds), np.array(all_labels), np.array(all_probs)

    def evaluate(self, model_name: str = "model") -> Dict[str, Any]:
        """
        Full evaluation: computes metrics and saves CSV + JSON report.

        Args:
            model_name: Identifier used for filenames.

        Returns:
            Dictionary containing accuracy, precision, recall, f1, and roc_auc.
        """
        preds, labels, probs = self.get_predictions()

        acc = float(accuracy_score(labels, preds))
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="weighted", zero_division=0
        )

        metrics: Dict[str, Any] = {
            "accuracy": acc,
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        }

        # ROC AUC for binary classification
        if len(self.class_names) == 2 and probs.shape[1] >= 2:
            try:
                auc = float(roc_auc_score(labels, probs[:, 1]))
                metrics["roc_auc"] = auc
            except Exception:
                pass

        cm = confusion_matrix(labels, preds).tolist()
        metrics["confusion_matrix"] = cm

        clf_report = classification_report(
            labels, preds, target_names=self.class_names, output_dict=True, zero_division=0
        )
        metrics["classification_report"] = clf_report

        print(f"\n================ Evaluation Results ({model_name}) ================")
        print(f"Accuracy:  {acc * 100:.2f}%")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        if "roc_auc" in metrics:
            print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
        print("Confusion Matrix:")
        print(np.array(cm))
        print("================================================================")

        # Save predictions CSV exactly matching notebook format
        results_df = pd.DataFrame({
            "True_Labels": labels,
            "Predicted_Labels": preds,
            "Probability_Class_0": [p[0] for p in probs],
            "Probability_Class_1": [p[1] for p in probs] if probs.shape[1] > 1 else [0.0] * len(probs),
        })

        csv_path = os.path.join(self.results_dir, f"test_predictions_{model_name}.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"[INFO] Test predictions saved to: {csv_path}")

        # Save metrics JSON
        json_path = os.path.join(self.results_dir, f"metrics_{model_name}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[INFO] Evaluation metrics saved to: {json_path}")

        return metrics
