# @title Gated AdaLN
import torch
import torch.nn as nn
import torch.nn.functional as F

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

# class LayerNorm2d(nn.LayerNorm):
class LayerNorm2d(nn.RMSNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x

class GatedAdaLN(nn.Module):
    def __init__(self, d_model, cond_dim=None):
        super().__init__()
        self.cond_dim = cond_dim
        self.adaLN = nn.Sequential(
            nn.SiLU(), zero_module(nn.Linear(d_model, 2 * d_model))
            # nn.SiLU(), zero_module(nn.Linear(d_model, 3 * d_model))
            # nn.SiLU(), zero_module(nn.Conv2d(d_model, 2 * d_model, 1))
            # nn.SiLU(), zero_module(nn.Conv2d(d_model, 3 * d_model, 1))
        )
        self.mlp = nn.Sequential(
            nn.SiLU(), zero_module(nn.Linear(d_model, d_model))
            # nn.SiLU(), zero_module(nn.Conv2d(d_model, d_model, 1))
        )
        self.norm = nn.LayerNorm(d_model)
        # self.norm = LayerNorm2d(d_model, elementwise_affine=False, eps=1e-6)
        # self.norm = LayerNorm2d(d_model)
        # self.norm = F.normalize
        # self.norm = nn.GroupNorm(1, d_model)

    def forward(self, x, cond=None): # [b,t,d] / [b,c,h,w]
        # if self.cond_dim==None: cond=x # is self attn
        scale, shift = self.adaLN(x).chunk(2, dim=-1) # for btd
        # # scale, shift = self.adaLN(x).chunk(2, dim=1) # for bchw
        x = x + (1 + scale) * self.norm(x) + shift

        # scale, shift, gate = self.adaLN(x).chunk(3, dim=-1)
        # scale, shift, gate = self.adaLN(cond)[...,None,None].chunk(3, dim=-1)
        # scale, shift, gate = self.adaLN(x).chunk(3, dim=1)

        # gate = torch.sigmoid(gate)
        # x = gate * ((1 + scale) * self.norm(x) + shift) + (1 - gate) * x

        # x = x + gate * self.mlp((1 + scale) * self.norm(x) + shift) # https://github.com/OliverRensu/FlowAR/blob/main/models/flowar.py#L157
        # x = gate * self.mlp((1 + scale) * self.norm(x) + shift)

        return x

# dim = 64
# model = GatedAdaLN(dim)

# x = torch.randn(2, dim)
# # x = torch.randn(2, dim, 7,9)
# out = model(x)
# print(out.shape)  # Should be (32, 64)
