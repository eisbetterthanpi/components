# @title efficientvit nn/ops.py down
# https://github.com/mit-han-lab/efficientvit/blob/master/efficientvit/models/nn/ops.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SameCh(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # print('samech', in_ch, out_ch, in_ch//out_ch, out_ch//in_ch)
        # if out_ch//in_ch > 1: self.func = lambda x: x.repeat_interleave(out_ch//in_ch, dim=1) # [b,i,h,w] -> [b,o,h,w]
        # elif in_ch//out_ch > 1:
        #     self.func = lambda x: torch.unflatten(x, 1, (out_ch, in_ch//out_ch)).mean(dim=2) # [b,i,h,w] -> [b,o,i/o,h,w] -> [b,o,h,w]
        # elif in_ch==out_ch: self.func = lambda x: x
        if in_ch>out_ch: # ave
            # self.func = lambda x: F.interpolate(x.flatten(2).transpose(1,2), size=out_ch, mode='nearest-exact').transpose(1,2).unflatten(-1,(x.shape[-2:])) # pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            self.func = lambda x: F.adaptive_avg_pool1d(x.flatten(2).transpose(1,2), out_ch).transpose(1,2).unflatten(-1,(x.shape[-2:])) # https://pytorch.org/docs/stable/nn.html#pooling-layers
            # self.func = lambda x: F.adaptive_max_pool1d(x.flatten(2).transpose(1,2), out_ch).transpose(1,2).unflatten(-1,(x.shape[-2:])) # https://pytorch.org/docs/stable/nn.html#pooling-layers
        elif in_ch<out_ch: # nearest
            self.func = lambda x: F.interpolate(x.flatten(2).transpose(1,2), size=out_ch, mode='nearest-exact').transpose(1,2).unflatten(-1,(x.shape[-2:])) # pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            # self.func = lambda x: F.adaptive_avg_pool1d(x.flatten(2).transpose(1,2), out_ch).transpose(1,2).unflatten(-1,(x.shape[-2:])) # https://pytorch.org/docs/stable/nn.html#pooling-layers
            # self.func = lambda x: F.adaptive_max_pool1d(x.flatten(2).transpose(1,2), out_ch).transpose(1,2).unflatten(-1,(x.shape[-2:])) # https://pytorch.org/docs/stable/nn.html#pooling-layers
        else:
            print('SameCh err', in_ch, out_ch)
    def forward(self, x): return self.func(x) # [b,c,h,w] -> [b,o,h,w]

class PixelShortcut(nn.Module):
    def __init__(self, in_ch, out_ch, r=1):
        super().__init__()
        self.r = r
        r = max(r, int(1/r))
        if self.r>1: self.net = nn.Sequential(SameCh(in_ch, out_ch*r**2), nn.PixelShuffle(r)) #
        elif self.r<1: self.net = nn.Sequential(nn.PixelUnshuffle(r), SameCh(in_ch*r**2, out_ch)) #
        else: self.net = SameCh(in_ch, out_ch)
    def forward(self, x): return self.net(x) # [b,c,h,w] -> [b,o,r*h,r*w]

def init_conv(conv, out_r=1, in_r=1):
    o, i, h, w = conv.weight.shape
    conv_weight = torch.empty(o//out_r**2, i//in_r**2, h, w)
    nn.init.kaiming_uniform_(conv_weight)
    conv.weight.data.copy_(conv_weight.repeat_interleave(out_r**2, dim=0).repeat_interleave(in_r**2, dim=1))
    if conv.bias is not None: nn.init.zeros_(conv.bias)
    return conv

class PixelShuffleConv(nn.Module):
    def __init__(self, in_ch, out_ch=None, kernel=3, r=1):
        super().__init__()
        self.r = r
        r = max(r, int(1/r))
        out_ch = out_ch or in_ch
        if self.r>1: self.net = nn.Sequential(nn.Conv2d(in_ch, out_ch*r**2, kernel, 1, kernel//2), nn.PixelShuffle(r)) # PixelShuffle: [b,c*r^2,h,w] -> [b,c,h*r,w*r] # upscale by upscale factor r # https://arxiv.org/pdf/1609.05158v2
        elif self.r<1: self.net = nn.Sequential(nn.PixelUnshuffle(r), nn.Conv2d(in_ch*r**2, out_ch, kernel, 1, kernel//2)) # PixelUnshuffle: [b,c,h*r,w*r] -> [b,c*r^2,h,w]
        # elif self.r<1: self.net = nn.Sequential(nn.PixelUnshuffle(r), nn.Conv2d(in_ch*r**2, out_ch, kernel, 1, kernel//2, bias=False)) # PixelUnshuffle: [b,c,h*r,w*r] -> [b,c*r^2,h,w]

        # if self.r>1: self.net = nn.Sequential(init_conv(nn.Conv2d(in_ch, out_ch*r**2, kernel, 1, kernel//2), out_r=r), nn.PixelShuffle(r)) # PixelShuffle: [b,c*r^2,h,w] -> [b,c,h*r,w*r] # upscale by upscale factor r # https://arxiv.org/pdf/1609.05158v2
        # elif self.r<1: self.net = nn.Sequential(nn.PixelUnshuffle(r), init_conv(nn.Conv2d(in_ch*r**2, out_ch, kernel, 1, kernel//2), in_r=r)) # PixelUnshuffle: [b,c,h*r,w*r] -> [b,c*r^2,h,w]
        else: self.net = nn.Conv2d(in_ch, out_ch, kernel, 1, kernel//2)
        # self.net.apply(self.init_conv_)

    def forward(self, x):
        return self.net(x)


class UpDownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, r=1, kernel=3):
        super().__init__()
        self.block = PixelShuffleConv(in_ch, out_ch, kernel=kernel, r=r)
        self.shortcut_block = PixelShortcut(in_ch, out_ch, r=r)
    def forward(self, x):
        # print('UpDownBlock', x.shape, self.block(x).shape, self.shortcut_block(x).shape)
        return self.block(x) + self.shortcut_block(x)

# model = UpDownBlock(3, 24, r=1/2)#.to(device)
# model = UpDownBlock(width_list[0], out_ch, r=2)

# x = torch.rand(64, 3, 32, 32, device=device)
x = torch.rand(2, 3, 32, 32)
# out = model(x)

# print(out.shape)
