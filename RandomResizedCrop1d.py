# @title RandomResizedCrop1d
import torch
import torch.nn as nn
import torch.nn.functional as F

class RandomResizedCrop1d(nn.Module):
    def __init__(self, size, scale=(.8,1.)): # output seq len, scale of seq to crop
        super().__init__()
        self.size, self.scale = size, scale

    def forward(self, x): # [batch, seq, dim]
        x = x.transpose(-2,-1)
        crop = torch.rand(1) * (self.scale[1] - self.scale[0]) + self.scale[0]
        pos = torch.rand(1) * (1 - crop)
        left, right = int(pos*x.shape[-1]), int((pos+crop)*x.shape[-1])
        # x = F.interpolate(x[...,left:right], size=x.shape[-1], mode='linear')
        x = F.adaptive_avg_pool1d(x[...,left:right], x.shape[-1]) # https://pytorch.org/docs/stable/nn.html#pooling-layers
        return x.transpose(-2,-1)

# transform = RandomResizedCrop1d(10)
# x = torch.randn(1,10,3)
# print(x)
# out = transform(x)
# print(out)
