"""
Training pipeline and Trainer class for DenseNet and ResNet.
"""

import os
import json
from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """
    Handles model training, validation, metric tracking, and checkpointing.
    Preserves exact logic from the project notebooks while adding MLOps features.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None,
        epochs: int = 20,
        save_dir: str = "models",
        model_name: str = "densenet201",
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.save_dir = save_dir
        self.model_name = model_name

        # Loss function (CrossEntropyLoss matching notebook classification)
        self.criterion = criterion or nn.CrossEntropyLoss()

        # Optimizer (Adam lr=0.001 matching notebooks)
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=0.001)

        os.makedirs(self.save_dir, exist_ok=True)

        self.history: Dict[str, list] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def train_epoch(self) -> Tuple[float, float]:
        """Runs a single training epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Zero parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Backward pass & optimize
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100.0 * correct / max(total, 1):.2f}%",
            })

        train_loss = running_loss / len(self.train_loader)
        train_acc = 100.0 * correct / max(total, 1)
        return train_loss, train_acc

    def validate(self) -> Tuple[float, float]:
        """Runs validation on the validation set."""
        if not self.val_loader:
            return 0.0, 0.0

        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = running_loss / len(self.val_loader)
        val_acc = 100.0 * correct / max(total, 1)
        return val_loss, val_acc

    def fit(self) -> Dict[str, list]:
        """
        Executes full training across all epochs with checkpointing.
        """
        best_val_acc = 0.0
        best_model_path = os.path.join(self.save_dir, f"{self.model_name}_best.pth")
        latest_model_path = os.path.join(self.save_dir, f"{self.model_name}.pth")

        print(f"\n[INFO] Starting training for {self.epochs} epochs on device: {self.device}")
        print(f"[INFO] Model: {self.model_name}")

        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_epoch()
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            msg = f"Epoch [{epoch+1}/{self.epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%"

            if self.val_loader:
                val_loss, val_acc = self.validate()
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                msg += f" | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"

                # Checkpoint best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(self.model.state_dict(), best_model_path)
                    msg += " [Best Checkpoint Saved]"
            else:
                # If no val_loader, save after each epoch if training improves
                torch.save(self.model.state_dict(), latest_model_path)

            print(msg)

        # Always save final model state_dict
        torch.save(self.model.state_dict(), latest_model_path)
        print(f"[INFO] Final model saved to: {latest_model_path}")

        # Save training history JSON
        history_path = os.path.join(self.save_dir, f"{self.model_name}_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        print(f"[INFO] Training history saved to: {history_path}")

        return self.history
