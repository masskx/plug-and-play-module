"""Fixture: a mix of clean modules, scaffolding and a whole model."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelGate(nn.Module):
    """Squeeze-style channel gate: (B, C, H, W) -> (B, C, H, W)."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.fc = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, channels), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.fc(x.mean(dim=(2, 3))).view(b, c, 1, 1)
        return x * w


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True).values
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class ChannelSpatialBlock(ChannelGate):
    """Subclasses a local module; still a module through inheritance."""

    def __init__(self, channels: int):
        super().__init__(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(super().forward(x))


def focal_loss(logits, targets, gamma: float = 2.0):
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


class SegTrainer:
    def __init__(self, cfg):
        self.cfg = cfg

    def train(self):
        raise NotImplementedError


class SegDataset(torch.utils.data.Dataset):
    pass


class TinyResNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3)
        self.gate = ChannelGate(8)
        self.head = nn.Linear(8, num_classes)

    def forward(self, x):
        x = self.gate(self.stem(x))
        return self.head(x.mean(dim=(2, 3)))


def tiny_conv_resnet(num_classes: int = 10):
    return TinyResNet(num_classes)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


if __name__ == "__main__":
    x = torch.randn(2, 16, 8, 8)
    print(ChannelSpatialBlock(16)(x).shape)
