"""
Model factory for instantiating and loading DenseNet and ResNet architectures.
Ensures seamless loading of pre-trained checkpoint weights.
"""

import os
from typing import Optional, Union, Dict, Any
import torch
import torch.nn as nn

from src.models.densenet import DenseNet, build_densenet, model_parameters
from src.models.resnet import ResNet, build_resnet, ResNet18, resnet_configurations


def get_model(
    architecture: str = "densenet201",
    num_classes: int = 2,
    in_channels: int = 3,
    weights_path: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> nn.Module:
    """
    Instantiate DenseNet or ResNet and optionally load pre-trained weights.

    Args:
        architecture: Architecture name ('densenet201', 'resnet18', etc.).
        num_classes: Number of output target classes (default: 2).
        in_channels: Input image channels (default: 3).
        weights_path: Path to .pth checkpoint file to load.
        device: Device to place the model on ('cuda', 'cpu', etc.).

    Returns:
        Instantiated nn.Module ready for training or evaluation/inference.
    """
    arch = architecture.lower().strip()

    # DenseNet family
    if "dense" in arch:
        variant = arch if arch in model_parameters else "densenet201"
        model = build_densenet(variant=variant, num_classes=num_classes, in_channels=in_channels)

    # ResNet family
    elif "res" in arch:
        variant = arch if arch in resnet_configurations else "resnet18"
        model = build_resnet(variant=variant, num_classes=num_classes)

    else:
        raise ValueError(
            f"Unsupported architecture '{architecture}'. "
            f"Supported options: DenseNet ({list(model_parameters.keys())}) "
            f"or ResNet ({list(resnet_configurations.keys())})."
        )

    # Load weights if provided
    if weights_path:
        load_model_weights(model, weights_path, device=device)

    if device:
        model = model.to(device)

    return model


def load_model_weights(
    model: nn.Module,
    weights_path: str,
    device: Optional[Union[str, torch.device]] = None,
) -> nn.Module:
    """
    Safely load weights from a checkpoint file into a model.
    Handles DataParallel 'module.' prefix stripping and various checkpoint dictionary formats.

    Args:
        model: PyTorch model instance.
        weights_path: Path to .pth file.
        device: Target device for loading.

    Returns:
        The model with loaded weights.
    """
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Weight file not found at: {weights_path}")

    map_location = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(weights_path, map_location=map_location)

    # If full checkpoint dictionary, extract state_dict
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        # Full model instance was saved directly
        if isinstance(checkpoint, nn.Module):
            state_dict = checkpoint.state_dict()
        else:
            state_dict = checkpoint

    # Clean DataParallel 'module.' prefix if present
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned_state_dict[k[7:]] = v
        else:
            cleaned_state_dict[k] = v

    model.load_state_dict(cleaned_state_dict, strict=True)
    print(f"Successfully loaded checkpoint weights from: {weights_path}")
    return model
