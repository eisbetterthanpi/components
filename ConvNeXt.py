# @title ConvNeXt
import torch
import torch.nn as nn
import torch.nn.functional as F
# from timm.models.layers import trunc_normal_, DropPath
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# class LayerNorm2d(nn.LayerNorm):
class LayerNorm2d(nn.RMSNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x

# https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/utils.py
class GRN(nn.Module):
    # """ GRN (Global Response Normalization) layer"""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x): # [b,c,h,w]
        # print('GRN', x.shape)
        Gx = torch.norm(x, p=2, dim=(1,2,3), keepdim=True)
        # print('GRN', Gx.shape)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        # print('GRN', Nx.shape)
        # print(self.gamma.shape ,(x * Nx).shape , self.beta.shape , x.shape)
        return self.gamma * (x * Nx) + self.beta + x

# https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py
class ConvNeXtV2Block(nn.Module):
    """ ConvNeXtV2 Block """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.gamma = nn.Parameter(1e-6 * torch.ones((dim)), requires_grad=True)
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim), LayerNorm2d(dim, eps=1e-6),
            nn.Conv2d(dim, 4 * dim, 1), nn.GELU(), #GRN(4 * dim),
            nn.Conv2d(4 * dim, dim, 1),# self.scale, DropPath(drop_path) if drop_path > 0. else nn.Identity()
        )

    def scale(self, x): return self.gamma * x
    def forward(self, x):
        return x + self.conv(x)

model = ConvNeXtV2Block(32).to(device)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
x = torch.rand((2,32,32,32), device=device)
out = model(x)
print(out.shape)
