"""
ResNet architecture definition matching the project's Jupyter notebooks exactly.
Preserves identical class names, attribute names, and module structure
to ensure compatibility with existing pre-trained .pth checkpoint files.
"""

from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

resnet_configurations: Dict[str, List[int]] = {
    "resnet18": [2, 2, 2, 2],
    "resnet34": [3, 4, 6, 3],
    "resnet101": [3, 4, 23, 3],
}


class ResidualBlock(nn.Module):
    """
    Basic Residual Block matching the notebooks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """
    ResNet architecture matching the notebook implementation.
    """

    def __init__(
        self,
        block: nn.Module = ResidualBlock,
        layers: Optional[List[int]] = None,
        num_classes: int = 2,
    ):
        super(ResNet, self).__init__()
        layers = layers if layers is not None else [2, 2, 2, 2]
        self.in_channels = 64

        # Initial Convolution + BatchNorm + ReLU + MaxPool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(
        self,
        block: nn.Module,
        out_channels: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def ResNet18(num_classes: int = 2) -> ResNet:
    """Instantiate ResNet18 as defined in the notebooks."""
    return ResNet(ResidualBlock, [2, 2, 2, 2], num_classes=num_classes)


def build_resnet(
    variant: str = "resnet18",
    num_classes: int = 2,
) -> ResNet:
    """Helper factory for instantiating ResNet variants."""
    variant_lower = variant.lower()
    if variant_lower in resnet_configurations:
        config = resnet_configurations[variant_lower]
    else:
        config = resnet_configurations["resnet18"]
    return ResNet(ResidualBlock, config, num_classes=num_classes)
