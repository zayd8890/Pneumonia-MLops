"""
DataLoader utilities for pneumonia classification.
Supports ImageFolder folder layout (train/val/test splits).
"""

import os
from typing import Tuple, Optional, List
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from src.data.dataset import get_transforms


def get_dataloaders(
    data_dir: Optional[str] = None,
    train_dir: Optional[str] = None,
    val_dir: Optional[str] = None,
    test_dir: Optional[str] = None,
    batch_size: int = 32,
    image_size: int = 224,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader], List[str]]:
    """
    Constructs PyTorch DataLoaders for training, validation, and testing.

    Args:
        data_dir: Root dataset folder containing 'train', 'val' (or 'validation'), and 'test' subfolders.
        train_dir: Explicit path to train folder.
        val_dir: Explicit path to validation folder.
        test_dir: Explicit path to test folder.
        batch_size: Number of images per batch (default: 32).
        image_size: Input resolution (default: 224).
        mean: Normalization channel means (default: ImageNet mean).
        std: Normalization channel standard deviations (default: ImageNet std).
        num_workers: DataLoader worker count.
        pin_memory: Enable pinned memory for faster GPU transfer.

    Returns:
        tuple of (train_loader, val_loader, test_loader, class_names)
    """
    mean = mean if mean is not None else [0.485, 0.456, 0.406]
    std = std if std is not None else [0.229, 0.224, 0.225]
    # Resolve directory paths
    if data_dir:
        if not train_dir:
            for candidate in ["train", "train_split"]:
                p = os.path.join(data_dir, candidate)
                if os.path.exists(p):
                    train_dir = p
                    break
        if not val_dir:
            for candidate in ["val", "val_split", "validation"]:
                p = os.path.join(data_dir, candidate)
                if os.path.exists(p):
                    val_dir = p
                    break
        if not test_dir:
            for candidate in ["test", "test_split"]:
                p = os.path.join(data_dir, candidate)
                if os.path.exists(p):
                    test_dir = p
                    break

    train_transform = get_transforms(image_size, mean, std, split="train")
    eval_transform = get_transforms(image_size, mean, std, split="val")

    train_loader = None
    val_loader = None
    test_loader = None
    class_names = ["NORMAL", "PNEUMONIA"]

    if train_dir and os.path.exists(train_dir):
        train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
        class_names = train_dataset.classes
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
        )

    if val_dir and os.path.exists(val_dir):
        val_dataset = datasets.ImageFolder(root=val_dir, transform=eval_transform)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
        )

    if test_dir and os.path.exists(test_dir):
        test_dataset = datasets.ImageFolder(root=test_dir, transform=eval_transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
        )

    return train_loader, val_loader, test_loader, class_names
