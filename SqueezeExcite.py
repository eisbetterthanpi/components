# @title squeeze excite
import torch
import torch.nn as nn

class SqueezeExcite(nn.Module):
    def __init__(self, d_model, red=16):
        super().__init__()
        self.lin = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # [b,c,1,1]
            nn.Conv2d(d_model, d_model//red, 1), nn.ReLU(),
            nn.Conv2d(d_model//red, d_model, 1), nn.Sigmoid()
        )

    def forward(self, x): # [b,c,h,w]
        return x * self.lin(x)
