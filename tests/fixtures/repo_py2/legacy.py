import torch.nn as nn


class OldBlock(nn.Module):
    def forward(self, x):
        print "python 2 print statement"
        return x
