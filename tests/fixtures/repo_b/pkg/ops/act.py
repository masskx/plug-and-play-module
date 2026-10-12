import torch.nn as nn
import torch.nn.functional as F


class HSwish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3.0) / 6.0
