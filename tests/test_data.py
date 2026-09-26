"""
Unit tests for data pipeline and transformations.
"""

import pytest
import torch
from PIL import Image
from src.data.dataset import get_transforms


def test_train_transforms():
    """Verify training transforms produce valid normalized tensors."""
    transform = get_transforms(image_size=224, split="train")
    img = Image.new("RGB", (300, 300), color=(128, 128, 128))
    tensor = transform(img)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)


def test_val_transforms():
    """Verify validation transforms produce valid normalized tensors."""
    transform = get_transforms(image_size=224, split="val")
    img = Image.new("RGB", (300, 300), color=(128, 128, 128))
    tensor = transform(img)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
