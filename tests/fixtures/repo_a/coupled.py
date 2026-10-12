from mmcv.cnn import build_norm_layer
from mmdet.models.builder import NECKS

from .utils import make_divisible

import torch.nn as nn


@NECKS.register_module()
class CoupledBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c = make_divisible(cfg.channels, 8)
        self.norm = build_norm_layer(cfg.norm_cfg, c)[1]
        self.proj = nn.Conv2d(c, c, 1)

    def forward(self, x):
        return self.proj(self.norm(x))
