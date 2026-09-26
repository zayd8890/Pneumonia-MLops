"""
Unit tests for DenseNet and ResNet model architectures.
"""

import pytest
import torch
from src.models.densenet import DenseNet, build_densenet, model_parameters
from src.models.resnet import ResNet, build_resnet, ResNet18, resnet_configurations
from src.models.model_factory import get_model


def test_densenet_instantiation():
    """Verify DenseNet instantiates and handles forward pass with proper output shape."""
    model = build_densenet(variant="densenet121", num_classes=2)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (2, 2), f"Expected shape (2, 2), got {output.shape}"


def test_densenet_variants():
    """Verify all DenseNet parameter configurations work."""
    for variant in model_parameters.keys():
        model = build_densenet(variant=variant, num_classes=2)
        assert isinstance(model, DenseNet)


def test_resnet_instantiation():
    """Verify ResNet instantiates and handles forward pass with proper output shape."""
    model = ResNet18(num_classes=2)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (2, 2), f"Expected shape (2, 2), got {output.shape}"


def test_resnet_variants():
    """Verify all ResNet variants instantiate correctly."""
    for variant in resnet_configurations.keys():
        model = build_resnet(variant=variant, num_classes=2)
        assert isinstance(model, ResNet)


def test_model_factory():
    """Verify model factory instantiates both architectures properly."""
    dense_model = get_model(architecture="densenet201", num_classes=2)
    assert isinstance(dense_model, DenseNet)

    res_model = get_model(architecture="resnet18", num_classes=2)
    assert isinstance(res_model, ResNet)
