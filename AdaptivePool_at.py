# @title AdaptivePool_at
import torch
import torch.nn as nn

class AdaptivePool_at(nn.AdaptiveAvgPool1d): # AdaptiveAvgPool1d AdaptiveMaxPool1d
    def __init__(self, dim=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim=dim
    def forward(self, x):
        x = x.transpose(self.dim,-1)
        shape = x.shape
        return super().forward(x.flatten(0,-2)).unflatten(0, shape[:-1]).transpose(self.dim,-1)

# x = torch.rand(2,3,4,4)
# ch_pool = AdaptivePool_at(1, output_size=5)
# print(ch_pool(x).shape)
