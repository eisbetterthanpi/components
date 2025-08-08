# @title StableNorm
import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class StableNorm(nn.Module): # https://openreview.net/pdf?id=lkRjnNW0gb
    def __init__(self, ndim, alpha=.375):
        super().__init__()
        self.ndim, self.alpha = ndim, alpha
        self.weight = nn.Parameter(torch.ones(1,1,ndim))

    def forward(self, x): # [b,t,d]
        x_norm = torch.norm(x, dim=2, keepdim=True) + 1e-8
        x = (self.ndim**self.alpha)*x/x_norm
        return self.weight*x

b,t,d = 2,7,4
x = torch.rand(b,t,d)
norm = nn.RMSNorm(d) # LayerNorm RMSNorm
norm1 = StableNorm(d)
out = norm(x)
print(out)
out = norm1(x)
print(out)
