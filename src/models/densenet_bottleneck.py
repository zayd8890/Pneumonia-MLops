"""
Bottleneck-style DenseNet, ported unchanged from notebooks/Copy_of_Untitled0.ipynb (cell 17).

Its layer names (conv1 = Sequential, features.dense_block_N..., bn, fc) differ from
src/models/densenet.py, so checkpoints trained with this notebook (e.g. the
'densenet121_*_scratch.pth' files) only load into this class.
"""

import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(Bottleneck, self).__init__()
        inner_channels = 4 * growth_rate
        self.bottleneck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inner_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, growth_rate, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        return torch.cat([x, self.bottleneck(x)], 1)


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.transition(x)


class DenseNetBottleneck(nn.Module):
    def __init__(self, block, nblocks, growth_rate=32, reduction=0.5, num_classes=2):
        super(DenseNetBottleneck, self).__init__()
        self.growth_rate = growth_rate
        inner_channels = 2 * growth_rate

        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, inner_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Dense blocks
        self.features = nn.Sequential()
        num_channels = inner_channels
        for i, num_layers in enumerate(nblocks):
            self.features.add_module(f"dense_block_{i}", self._make_dense_block(block, num_layers, num_channels))
            num_channels += num_layers * growth_rate
            if i != len(nblocks) - 1:
                out_channels = int(num_channels * reduction)
                self.features.add_module(f"transition_{i}", Transition(num_channels, out_channels))
                num_channels = out_channels

        # Final layers
        self.bn = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_channels, num_classes)

    def _make_dense_block(self, block, num_layers, in_channels):
        layers = []
        for _ in range(num_layers):
            layers.append(block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def densenet121_bottleneck(num_classes: int = 2) -> DenseNetBottleneck:
    return DenseNetBottleneck(Bottleneck, [6, 12, 24, 16], growth_rate=32, num_classes=num_classes)
