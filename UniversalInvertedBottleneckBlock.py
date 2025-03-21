# @title UniversalInvertedBottleneckBlock
import torch
import torch.nn as nn
import torch.nn.functional as F

class UIB(nn.Module):
    def __init__(self, in_ch, out_ch=None, mult=4):
        super().__init__()
        act = nn.SiLU()
        out_ch = out_ch or in_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, 1, 3//2, groups=in_ch, bias=False), nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, mult*in_ch, 1, bias=False), nn.BatchNorm2d(mult*in_ch), act,
            nn.Conv2d(mult*in_ch, mult*in_ch, 3, 1, 3//2, groups=mult*in_ch, bias=False), nn.BatchNorm2d(mult*in_ch), act,
            nn.Conv2d(mult*in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch),
        )
        self.gamma = nn.Parameter(1e-5 * torch.ones(out_ch, 1, 1))

    def forward(self,x):
        x = self.conv(x) * self.gamma
        return x

in_ch, out_ch = 3, 3
model = UIB(in_ch, out_ch)
x = torch.randn(2, in_ch, 7,9)
out = model(x)
print(out.shape)
