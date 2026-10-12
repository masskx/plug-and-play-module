import torch
import torch.nn as nn

from .ops.act import HSwish
from .ops.util import make_divisible as md


class DWBlock(nn.Module):
    """Depthwise-separable block that leans on helpers from sibling files."""

    def __init__(self, channels: int):
        super().__init__()
        hidden = md(channels * 2)
        self.expand = nn.Conv2d(channels, hidden, 1)
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.act = HSwish()
        self.project = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        return x + self.project(self.act(self.dw(self.act(self.expand(x)))))


class Unrelated(nn.Module):
    def forward(self, x):
        return x


if __name__ == "__main__":
    x = torch.randn(1, 16, 8, 8)
    print(DWBlock(16)(x).shape)
