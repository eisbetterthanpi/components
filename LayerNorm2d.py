import torch
import torch.nn as nn

class LayerNorm2d(nn.RMSNorm): # LayerNorm RMSNorm
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, x): return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

# class LayerNorm2d(nn.LayerNorm):
class LayerNorm2d(nn.RMSNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x
