Adaptive Fourier Neural Operator (AFNO)
# @title AFNO2D
# Adaptive Frequency Filters As Efficient Global Token Mixers https://arxiv.org/pdf/2307.14008
# https://github.com/microsoft/TokenMixers/blob/main/Adaptive%20Frequency%20Filters/affnet/modules/aff_block.py#L62
# https://github.com/NVlabs/AFNO-transformer/blob/master/afno/afno2d.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AFNO2D(nn.Module):
    def __init__(self, d_model, n_blocks=8, h_factor=1):
        super().__init__()
        self.n_blocks = n_blocks # n_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
        self.block_size = d_model // self.n_blocks
        scale = 0.02

        self.w1 = nn.Parameter(scale * torch.randn(2, self.n_blocks, self.block_size, self.block_size * h_factor))
        self.b1 = nn.Parameter(scale * torch.randn(2, self.n_blocks, self.block_size * h_factor))
        self.w2 = nn.Parameter(scale * torch.randn(2, self.n_blocks, self.block_size * h_factor, self.block_size))
        self.b2 = nn.Parameter(scale * torch.randn(2, self.n_blocks, self.block_size))

    # torch.amp.autocast(enabled=False)
    def forward(self, x): # [b,c,h,w]
        # bias = x
        dtype = x.dtype
        B, C, H, W = x.shape # asert C % d_model == 0
        x = x.float().permute(0,2,3,1) # [b, h, w, c]

        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho") # [b, h, w//2+1, c]
        origin_ffted = x
        x = x.unflatten(-1, (self.n_blocks, self.block_size)) # [b, h, w//2+1, n_blocks, block_size]

        o1_real = F.relu((x.real.unsqueeze(-2) @ self.w1[0] - x.imag.unsqueeze(-2) @ self.w1[1]).squeeze(-2) + self.b1[0]) # [b, h, w//2+1, n_blocks, block_size] @ [n_blocks, block_size, block_size * h_factor] -> [b, h, w//2+1, n_blocks, block_size * h_factor] # + [n_blocks, block_size * hfactor]
        o1_imag = F.relu((x.imag.unsqueeze(-2) @ self.w1[0] - x.real.unsqueeze(-2) @ self.w1[1]).squeeze(-2) + self.b1[1])
        o2_real = F.relu((o1_real.unsqueeze(-2) @ self.w2[0] - o1_imag.unsqueeze(-2) @ self.w2[1]).squeeze(-2) + self.b2[0]) # [n_blocks, block_size * h_factor, block_size]
        o2_imag = F.relu((o1_imag.unsqueeze(-2) @ self.w2[0] - o1_real.unsqueeze(-2) @ self.w2[1]).squeeze(-2) + self.b2[1])

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=0.01)
        x = torch.view_as_complex(x)
        x = x.flatten(-2,-1) # [b, h, w, c]

        x = x * origin_ffted
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
        x = x.type(dtype).permute(0,3,1,2) # [b, c, h, w]
        return x #+ bias

b,c,h,w = 2,16,9,9
mix = AFNO2D(c, 4)
x = torch.rand(b,c,h,w)
out = mix(x)
print(out.shape)
# print(out)
