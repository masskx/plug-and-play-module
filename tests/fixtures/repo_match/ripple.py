import torch
import torch.nn as nn


class RippleAttention(nn.Module):
    """Channel recalibration: (B, C, H, W) -> (B, C, H, W)."""

    def __init__(self, k: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x):
        z = x.mean(dim=(2, 3))                       # (B, C)
        w = torch.sigmoid(self.conv(z.unsqueeze(1))).squeeze(1)
        return x * w[:, :, None, None] + x


if __name__ == "__main__":
    x = torch.randn(2, 16, 8, 8, requires_grad=True)
    y = RippleAttention()(x)
    y.sum().backward()
    print("input", tuple(x.shape), "output", tuple(y.shape))
