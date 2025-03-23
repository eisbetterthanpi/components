# @title UpDownBlock_me
import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelShuffleConv(nn.Module):
    def __init__(self, in_ch, out_ch=None, kernel=3, r=1):
        super().__init__()
        self.r = r
        r = max(r, int(1/r))
        out_ch = out_ch or in_ch
        # if self.r>1: self.net = nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, kernel, 2, kernel//2, output_padding=1))
        # if self.r>1: self.net = nn.Sequential(nn.Conv2d(in_ch, out_ch*r**2, kernel, 1, kernel//2), nn.PixelShuffle(r)) # PixelShuffle: [b,c*r^2,h,w] -> [b,c,h*r,w*r] # upscale by upscale factor r # https://arxiv.org/pdf/1609.05158v2
        if self.r>1: self.net = nn.Sequential(UIB(in_ch, out_ch*r**2, r=r), nn.PixelShuffle(r))
        # if self.r>1: self.net = nn.Sequential(ResBlock(in_ch, out_ch*r**2), nn.PixelShuffle(r))

        # elif self.r<1: self.net = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel, 2, kernel//2))
        # elif self.r<1: self.net = nn.Sequential(nn.PixelUnshuffle(r), nn.Conv2d(in_ch*r**2, out_ch, kernel, 1, kernel//2)) # PixelUnshuffle: [b,c,h*r,w*r] -> [b,c*r^2,h,w]
        elif self.r<1: self.net = nn.Sequential(nn.PixelUnshuffle(r), UIB(in_ch*r**2, out_ch, r=r))
        # elif self.r<1: self.net = nn.Sequential(nn.PixelUnshuffle(r), ResBlock(in_ch*r**2, out_ch))

    def forward(self, x):
        return self.net(x)

class UpDownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, r=1, kernel=3):
        super().__init__()
        # self.block = PixelShuffleConv(in_ch, out_ch, kernel=kernel, r=r)
        act = nn.SiLU()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_ch), act, PixelShuffleConv(in_ch, out_ch, kernel=kernel, r=r)
        )

    def forward(self, x): # [b,c,h,w]
        out = self.block(x)
        # shortcut = F.interpolate(x.unsqueeze(1), size=out.shape[1:], mode='nearest-exact').squeeze(1) # pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        shortcut = F.adaptive_avg_pool3d(x, out.shape[1:]) # https://pytorch.org/docs/stable/nn.html#pooling-layers
        # shortcut = F.adaptive_max_pool3d(x, out.shape[1:]) # https://pytorch.org/docs/stable/nn.html#pooling-layers
        # shortcut = F.adaptive_avg_pool3d(x, out.shape[1:]) if out.shape[1]>=x.shape[1] else F.adaptive_max_pool3d(x, out.shape[1:])
        return out + shortcut

# if out>in, inter=max=near. ave=ave
# if out<in, inter=ave. max=max

# in_ch, out_ch = 16,3
in_ch, out_ch = 3,16
# model = UpDownBlock(in_ch, out_ch, r=1/2).to(device)
model = UpDownBlock(in_ch, out_ch, r=2).to(device)

x = torch.rand(12, in_ch, 64,64, device=device)
out = model(x)

print(out.shape)
