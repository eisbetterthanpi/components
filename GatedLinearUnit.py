# @title Gated Linear Unit
import torch
import torch.nn as nn
import torch.nn.functional as F

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

class GLU(nn.Module): # https://arxiv.org/pdf/2002.05202
    def __init__(self, in_dim, d_model):
        super().__init__()
        self.lin0 = nn.Sequential(
            # nn.SiLU(), zero_module(nn.Linear(in_dim, 2*d_model))
            # nn.LayerNorm(in_dim), nn.SiLU(), zero_module(nn.Linear(in_dim, 2*d_model, bias=False))
            # nn.LayerNorm(in_dim), zero_module(nn.Linear(in_dim, 2*d_model, bias=False))
            zero_module(nn.Linear(in_dim, 2*d_model, bias=False))
        )
        self.lin1 = nn.Sequential(
            # nn.SiLU(), zero_module(nn.Linear(d_model, in_dim))
            zero_module(nn.Linear(d_model, in_dim, bias=False))
        )
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x): # [b,t,d]
        x0, x1 = self.lin0(x).chunk(2, dim=-1)
        # x = x + x0 * self.norm(x) + x1 # AdaLN
        # x = x + self.lin1(x0*self.norm(x)+x1) # AdaLN
        # x0, x1, x2 = self.lin0(x).chunk(3, dim=-1)
        # x = x + x0 * x1 # Bilinear
        # x = x + x0 * x1 + x2
        # x = x + x0.exp() * x1 + x2

        # x = self.lin1(x0*F.sigmoid(x1)) # GLU
        # x = self.lin1(x0*F.gelu(x1)) # GEGLU
        # x = self.lin1(x0*F.silu(x1)) # SwiGLU
        # x = self.lin1(x0*F.relu(x1)) # ReGLU
        # x = self.lin1(x0*x1.exp()) #
        x = self.lin1(x0*x1) # Bilinear
        return x



class SwiGLU(nn.Module): # https://arxiv.org/pdf/2002.05202
    def __init__(self, in_dim, d_model):
        super().__init__()
        self.lin0 = zero_module(nn.Linear(in_dim, 2*d_model, bias=False))
        self.lin1 = zero_module(nn.Linear(d_model, in_dim, bias=False))

    def forward(self, x): # [b,t,d]
        x0, x1 = self.lin0(x).chunk(2, dim=-1)
        return self.lin1(x0*F.silu(x1)) # SwiGLU


class AGeLU(nn.Module): # https://openreview.net/pdf?id=I8pdQLfR77
    def __init__(self):
        super().__init__()
        self.act = nn.GELU() # GELU SiLU
        self.coef = nn.Parameter(torch.randn(4))

    def forward(self, x): # [b,t,d]
        return self.coef[0] * self.act(self.coef[1] * x + self.coef[2]) + self.coef[3]


class AMLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.lin0 = nn.Sequential(
            nn.SiLU(), nn.Linear(d_model, 2*d_model)
        )
        self.lin1 = nn.Sequential(
            zero_module(nn.Linear(2*d_model, d_model))
        )
        self.act = nn.GELU() # GELU SiLU
        self.agelu0, self.agelu1 = AGeLU(), AGeLU()

    def forward(self, x): # [b,t,d]
        x0, x1 = self.lin0(x).chunk(2, dim=-1) # for btd
        x = torch.cat([self.agelu0(x0), self.agelu1(x1)], dim=-1)
        return self.lin1(x)

dim = 64
model = GLU(dim, int(3.5*dim))
# model = GLU(dim, dim)
# model = AMLP(dim)

# adaln:nal,ada,al < adaln:nal,ada <

x = torch.randn(2, 3,dim)
out = model(x)
print(out.shape)  # Should be (32, 64)
