# @title ConvNeXt
import torch
import torch.nn as nn
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class LayerNorm2d(nn.RMSNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x

def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training: return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders https://arxiv.org/pdf/2301.00808
# https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/utils.py#L105
class GRN(nn.Module): # GRN (Global Response Normalization) layer
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x): # [b,h,w,c]
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class GRN(nn.Module): # GRN (Global Response Normalization) layer
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x): # [b,c,h,w]
        Gx = torch.norm(x, p=2, dim=(-2,-1), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

# https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py#L14
class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim),
            LayerNorm2d(dim, eps=1e-6), nn.Conv2d(dim, 4*dim, 1), nn.GELU(),
            GRN(4*dim), nn.Conv2d(4*dim, dim, 1),
            DropPath(drop_path) if drop_path > 0. else nn.Identity()
        )

    def forward(self, x): # [b,c,h,w]
        return x + self.conv(x)

dim=3
model = ConvNeXtV2Block(dim, .1).to(device)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
x = torch.rand((2,dim,32,32), device=device)
out = model(x)
print(out.shape)
