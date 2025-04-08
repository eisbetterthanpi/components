# @title UpDownBlock_me
import torch
import torch.nn as nn
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PixelShuffleConv(nn.Module):
    def __init__(self, in_ch, out_ch=None, kernel=1, r=1):
        super().__init__()
        self.r = r
        r = max(r, int(1/r))
        out_ch = out_ch or in_ch
        if self.r>1: self.net = nn.Sequential(ResBlock(in_ch, out_ch*r**2, kernel), nn.PixelShuffle(r))
        # if self.r>1: self.net = nn.Sequential(Attention(in_ch, out_ch*r**2), nn.PixelShuffle(r))
# MaskUnitAttention(in_dim, d_model=16, n_heads=4, q_stride=None, nd=2)

        elif self.r<1: self.net = nn.Sequential(nn.PixelUnshuffle(r), ResBlock(in_ch*r**2, out_ch, kernel))
        # elif self.r<1: self.net = nn.Sequential(nn.PixelUnshuffle(r), Attention(in_ch*r**2, out_ch))
        elif in_ch != out_ch: self.net = ResBlock(in_ch*r**2, out_ch, kernel)
        else: self.net = lambda x: torch.zeros_like(x)

    def forward(self, x):
        return self.net(x)

def AdaptiveAvgPool_nd(n, *args, **kwargs): return [nn.Identity, nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d][n](*args, **kwargs)
def AdaptiveMaxPool_nd(n, *args, **kwargs): return [nn.Identity, nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d][n](*args, **kwargs)


class AdaptivePool_at(nn.AdaptiveAvgPool1d): # AdaptiveAvgPool1d AdaptiveMaxPool1d
    def __init__(self, dim=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim=dim
    def forward(self, x):
        x = x.transpose(self.dim,-1)
        shape = x.shape
        return super().forward(x.flatten(0,-2)).unflatten(0, shape[:-1]).transpose(self.dim,-1)


def adaptive_pool_at(x, dim, output_size, pool='avg'): # [b,c,h,w]
    x = x.transpose(dim,-1)
    shape = x.shape
    parent={'avg':F.adaptive_avg_pool1d, 'max':F.adaptive_max_pool1d}[pool]
    return parent(x.flatten(0,-2), output_size).unflatten(0, shape[:-1]).transpose(dim,-1)


class ZeroExtend():
    def __init__(self, dim=1, output_size=16):
        self.dim, self.out = dim, output_size
    def __call__(self, x): # [b,c,h,w]
        return torch.cat((x, torch.zeros(*x.shape[:self.dim], self.out - x.shape[self.dim], *x.shape[self.dim+1:])), dim=self.dim)

def make_pool_at(pool='avg', dim=1, output_size=5):
    parent={'avg':nn.AdaptiveAvgPool1d, 'max':nn.AdaptiveMaxPool1d}[pool]
    class AdaptivePool_at(parent): # AdaptiveAvgPool1d AdaptiveMaxPool1d
        def __init__(self, dim=1, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.dim=dim
        def forward(self, x):
            x = x.transpose(self.dim,-1)
            shape = x.shape
            return super().forward(x.flatten(0,-2)).unflatten(0, shape[:-1]).transpose(self.dim,-1)
    return AdaptivePool_at(dim, output_size=output_size)

class Shortcut():
    def __init__(self, dim=1, c=3, sp=(3,3), nd=2):
        self.dim = dim
        # self.ch_pool = make_pool_at(pool='avg', dim=dim, output_size=c)
        self.ch_pool = make_pool_at(pool='max', dim=dim, output_size=c)
        # self.ch_pool = ZeroExtend(dim, output_size=c) # only for out_dim>=in_dim
        # self.sp_pool = AdaptiveAvgPool_nd(nd, sp)
        self.sp_pool = AdaptiveMaxPool_nd(nd, sp)

    def __call__(self, x): # [b,c,h,w]
        x = self.sp_pool(x) # spatial first preserves spatial more?
        x = self.ch_pool(x)
        return x

class UpDownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7, r=1):
        super().__init__()
        act = nn.SiLU()
        self.r = r
        self.block = PixelShuffleConv(in_ch, out_ch, kernel=kernel, r=r)
        # self.block = nn.Sequential(
        #     nn.BatchNorm2d(in_ch), act, PixelShuffleConv(in_ch, out_ch, kernel=kernel, r=r)
        # )
        # if self.r>1: self.res_conv = nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, kernel, 2, kernel//2, output_padding=1))
        # if self.r>1: self.res_conv = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(in_ch, out_ch, kernel, 1, kernel//2) if in_ch != out_ch else nn.Identity())
        # if self.r>1: self.res_conv = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(in_ch, out_ch, kernel, 1, kernel//2) if in_ch != out_ch else nn.Identity())
        # if self.r>1: self.res_conv = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(in_ch, out_ch, kernel, 1, kernel//2) if in_ch != out_ch else nn.Identity())


        # elif self.r<1: self.res_conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel, 2, kernel//2))
        # elif self.r<1: self.res_conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel, 1, kernel//2) if in_ch != out_ch else nn.Identity(), nn.MaxPool2d(2,2))
        # elif self.r<1: self.res_conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel, 1, kernel//2) if in_ch != out_ch else nn.Identity(), nn.AvgPool2d(2,2))

        # else: self.res_conv = nn.Conv2d(in_ch, out_ch, kernel, 1, kernel//2) if in_ch != out_ch else nn.Identity()

    def forward(self, x): # [b,c,h,w]
        b, num_tok, c, *win = x.shape
        x = x.flatten(0,1)
        out = self.block(x)
        # # shortcut = F.interpolate(x.unsqueeze(1), size=out.shape[1:], mode='nearest-exact').squeeze(1) # pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        # shortcut = F.adaptive_avg_pool3d(x, out.shape[1:]) # https://pytorch.org/docs/stable/nn.html#pooling-layers
        # # shortcut = F.adaptive_max_pool3d(x, out.shape[1:]) # https://pytorch.org/docs/stable/nn.html#pooling-layers
        # # shortcut = F.adaptive_avg_pool3d(x, out.shape[1:]) if out.shape[1]>=x.shape[1] else F.adaptive_max_pool3d(x, out.shape[1:])
        # shortcut(x)
        shortcut = Shortcut(dim=1, c=out.shape[1], sp=out.shape[-2:], nd=2)(x)
        out = out + shortcut
        out = out.unflatten(0, (b, num_tok))
        return out

        # return out + shortcut + self.res_conv(x)
        # return out + self.res_conv(x)
        # return self.res_conv(x)

# if out>in, inter=max=ave=near.
# if out<in, inter=ave. max=max

# stride2
# interconv/convpool
# pixelconv
# pixeluib
# pixelres
# shortcut

# in_ch, out_ch = 16,3
in_ch, out_ch = 3,16
model = UpDownBlock(in_ch, out_ch, r=1/2).to(device)
# model = UpDownBlock(in_ch, out_ch, r=2).to(device)

x = torch.rand(12, in_ch, 64,64, device=device)
x = torch.rand(12, 2, in_ch, 64,64, device=device)
out = model(x)

print(out.shape)
