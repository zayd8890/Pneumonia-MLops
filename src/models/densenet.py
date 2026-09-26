"""
DenseNet architecture definition matching the project's Jupyter notebooks exactly.
Preserves identical class names, attribute names, and module structure
to ensure compatibility with existing pre-trained .pth checkpoint files.
"""

from typing import List, Dict, Union
import torch
import torch.nn as nn

# Growth rate and compression factor from notebooks
k = 32  # Growth rate
compression_factor = 0.5

# Model configurations for DenseNet variants
model_parameters: Dict[str, List[int]] = {
    "densenet121": [6, 12, 24, 16],
    "densenet169": [6, 12, 32, 32],
    "densenet201": [6, 12, 48, 32],
    "densenet264": [6, 12, 64, 48],
}


class DenseLayer(nn.Module):
    """
    Individual DenseLayer: BN -> Conv1x1 -> BN -> Conv3x3 -> Concat.
    """

    def __init__(self, in_channels: int):
        super(DenseLayer, self).__init__()
        self.BN1 = nn.BatchNorm2d(num_features=in_channels)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=4 * k,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.BN2 = nn.BatchNorm2d(num_features=4 * k)
        self.conv2 = nn.Conv2d(
            in_channels=4 * k,
            out_channels=k,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xin = x
        x = self.relu(self.BN1(x))
        x = self.conv1(x)
        x = self.relu(self.BN2(x))
        x = self.conv2(x)
        x = torch.cat([xin, x], dim=1)  # Concatenate input and output along channels
        return x


class DenseBlock(nn.Module):
    """
    DenseBlock composed of multiple DenseLayers.
    """

    def __init__(self, num_layers: int, in_channels: int):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(DenseLayer(in_channels + i * k))  # Add k channels per layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class TransitionLayer(nn.Module):
    """
    TransitionLayer: BN -> Conv1x1 -> AvgPool2d.
    Reduces feature map resolution and channel count.
    """

    def __init__(self, in_channels: int, compression_factor: float = compression_factor):
        super(TransitionLayer, self).__init__()
        self.BN = nn.BatchNorm2d(num_features=in_channels)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=int(in_channels * compression_factor),
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.BN(x)
        x = self.conv(x)
        x = self.avgpool(x)
        return x


class DenseNet(nn.Module):
    """
    Full DenseNet model matching the notebook implementation.
    """

    def __init__(
        self,
        densenet_variant: Union[str, List[int]] = "densenet201",
        in_channels: int = 3,
        num_classes: int = 2,
    ):
        """
        Args:
            densenet_variant: List of layer counts per block (e.g., [6, 12, 48, 32])
                             or a variant name (e.g. 'densenet201').
            in_channels: Input image channels (default: 3).
            num_classes: Number of output classes (default: 2 for binary classification).
        """
        super(DenseNet, self).__init__()

        # Resolve variant if string was passed
        if isinstance(densenet_variant, str):
            variant_name = densenet_variant.lower()
            if variant_name in model_parameters:
                layers_config = model_parameters[variant_name]
            else:
                layers_config = model_parameters["densenet201"]
        else:
            layers_config = densenet_variant

        # Initial convolution and pooling
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.BN1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # DenseBlocks and TransitionLayers
        self.blocks = nn.ModuleList()
        in_channels_block = 64  # Output channels after initial conv

        for i, num_layers in enumerate(layers_config):
            # Add DenseBlock
            self.blocks.append(DenseBlock(num_layers, in_channels_block))
            in_channels_block += num_layers * k  # Update input channels for next block

            # Add TransitionLayer (except after the last block)
            if i != len(layers_config) - 1:
                self.blocks.append(TransitionLayer(in_channels_block, compression_factor))
                in_channels_block = int(in_channels_block * compression_factor)

        # Final layers
        self.BN2 = nn.BatchNorm2d(num_features=in_channels_block)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels_block, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial convolution and pooling
        x = self.relu(self.BN1(self.conv1(x)))
        x = self.maxpool(x)

        # DenseBlocks and TransitionLayers
        for block in self.blocks:
            x = block(x)

        # Final layers
        x = self.relu(self.BN2(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def build_densenet(
    variant: str = "densenet201",
    num_classes: int = 2,
    in_channels: int = 3,
) -> DenseNet:
    """Helper factory for instantiating DenseNet."""
    variant_lower = variant.lower()
    if variant_lower in model_parameters:
        config = model_parameters[variant_lower]
    else:
        config = model_parameters["densenet201"]
    return DenseNet(config, in_channels=in_channels, num_classes=num_classes)
