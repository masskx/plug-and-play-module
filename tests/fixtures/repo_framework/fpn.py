from mmengine.model import BaseModule
from mmdet.registry import MODELS
from mmcv.cnn import ConvModule


@MODELS.register_module()
class FPN(BaseModule):
    """Real-world shape: framework base class, registry decorator, no keyword in the name."""

    def __init__(self, in_channels, out_channels, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.convs = [ConvModule(c, out_channels, 1) for c in in_channels]

    def forward(self, inputs):
        return tuple(conv(x) for conv, x in zip(self.convs, inputs))
