# @title UpDownBlock_me
import torch
import torch.nn as nn
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch=None, emb_dim=None, drop=0.):
        super().__init__()
        if out_ch==None: out_ch=in_ch
        act = nn.SiLU() #
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        # self.res_conv = zero_module(nn.Conv2d(in_ch, out_ch, 1)) if in_ch != out_ch else nn.Identity()

        # self.block = nn.Sequential( # best?
        #     nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), act,
        #     zero_module(nn.Conv2d(out_ch, out_ch, 3, padding=1)), nn.BatchNorm2d(out_ch), act,
        #     )
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_ch), act, nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), act, zero_module(nn.Conv2d(out_ch, out_ch, 3, padding=1)),
            )

    def forward(self, x, emb=None): # [b,c,h,w], [batch, emb_dim]
        return self.block(x) + self.res_conv(x)

class PixelShuffleConv(nn.Module):
    def __init__(self, in_ch, out_ch=None, kernel=3, r=1):
        super().__init__()
        self.r = r
        r = max(r, int(1/r))
        out_ch = out_ch or in_ch
        if self.r>1: self.net = nn.Sequential(ResBlock(in_ch, out_ch*r**2), nn.PixelShuffle(r))
        elif self.r<1: self.net = nn.Sequential(nn.PixelUnshuffle(r), ResBlock(in_ch*r**2, out_ch))

    def forward(self, x):
        return self.net(x)

class UpDownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7, r=1):
        super().__init__()
        self.block = PixelShuffleConv(in_ch, out_ch, kernel=kernel, r=r)

    def forward(self, x): # [b,c,h,w]
        out = self.block(x)
        shortcut = F.adaptive_avg_pool3d(x, out.shape[1:]) # https://pytorch.org/docs/stable/nn.html#pooling-layers
        return out + shortcut




class PixelShuffleConv(nn.Module):
    def __init__(self, in_ch, out_ch=None, kernel=3, r=1):
        super().__init__()
        self.r = r
        r = max(r, int(1/r))
        out_ch = out_ch or in_ch
        # if self.r>1: self.net = nn.Sequential(nn.Conv2d(in_ch, out_ch*r**2, kernel, 1, kernel//2), nn.PixelShuffle(r)) # PixelShuffle: [b,c*r^2,h,w] -> [b,c,h*r,w*r] # upscale by upscale factor r # https://arxiv.org/pdf/1609.05158v2
        # if self.r>1: self.net = nn.Sequential(UIB(in_ch, out_ch*r**2, kernel), nn.PixelShuffle(r))
        if self.r>1: self.net = nn.Sequential(ResBlock(in_ch, out_ch*r**2), nn.PixelShuffle(r))

        # elif self.r<1: self.net = nn.Sequential(nn.PixelUnshuffle(r), nn.Conv2d(in_ch*r**2, out_ch, kernel, 1, kernel//2)) # PixelUnshuffle: [b,c,h*r,w*r] -> [b,c*r^2,h,w]
        # elif self.r<1: self.net = nn.Sequential(nn.PixelUnshuffle(r), UIB(in_ch*r**2, out_ch, kernel))
        elif self.r<1: self.net = nn.Sequential(nn.PixelUnshuffle(r), ResBlock(in_ch*r**2, out_ch))

    def forward(self, x):
        return self.net(x)

class Interpolate(nn.Module):
    def __init__(self, scale_factor=2, mode="nearest-exact", **kwargs):
        super().__init__()
        self.kwargs = kwargs
    def forward(self, x):
        # return F.interpolate(x, scale_factor=2, mode="nearest-exact", **self.kwargs)
        return F.adaptive_avg_pool2d(x, (x.shape[2]*2, x.shape[3]*2))

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
        # if self.r>1: self.res_conv = nn.Sequential(Interpolate(), nn.Conv2d(in_ch, out_ch, kernel, 1, kernel//2) if in_ch != out_ch else nn.Identity())

        # elif self.r<1: self.res_conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel, 2, kernel//2))
        # elif self.r<1: self.res_conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel, 1, kernel//2) if in_ch != out_ch else nn.Identity(), nn.MaxPool2d(2,2))
        # elif self.r<1: self.res_conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel, 1, kernel//2) if in_ch != out_ch else nn.Identity(), nn.AvgPool2d(2,2))

        # else: self.res_conv = nn.Conv2d(in_ch, out_ch, kernel, 1, kernel//2) if in_ch != out_ch else nn.Identity()

    def forward(self, x): # [b,c,h,w]
        out = self.block(x)
        # shortcut = F.interpolate(x.unsqueeze(1), size=out.shape[1:], mode='nearest-exact').squeeze(1) # pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        shortcut = F.adaptive_avg_pool3d(x, out.shape[1:]) # https://pytorch.org/docs/stable/nn.html#pooling-layers
        # shortcut = F.adaptive_max_pool3d(x, out.shape[1:]) # https://pytorch.org/docs/stable/nn.html#pooling-layers
        # shortcut = F.adaptive_avg_pool3d(x, out.shape[1:]) if out.shape[1]>=x.shape[1] else F.adaptive_max_pool3d(x, out.shape[1:])
        return out + shortcut
        # return out + shortcut + self.res_conv(x)
        # return out + self.res_conv(x)
        # return self.res_conv(x)

# if out>in, inter=max=near. ave=ave
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
out = model(x)

print(out.shape)
