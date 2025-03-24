# @title UIB
import torch
import torch.nn as nn

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

class UIB(nn.Module):
    def __init__(self, in_ch, out_ch=None, kernel=3, mult=4):
        super().__init__()
        act = nn.SiLU()
        out_ch = out_ch or in_ch
        self.conv = nn.Sequential( # norm,act,conv
            nn.BatchNorm2d(in_ch), act, nn.Conv2d(in_ch, in_ch, kernel, 1, kernel//2, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch), act, nn.Conv2d(in_ch, mult*in_ch, 1, bias=False),
            nn.BatchNorm2d(mult*in_ch), act, nn.Conv2d(mult*in_ch, mult*in_ch, kernel, 1, kernel//2, groups=mult*in_ch, bias=False),
            nn.BatchNorm2d(mult*in_ch), act, zero_module(nn.Conv2d(mult*in_ch, out_ch, 1, bias=False)),
        )

    def forward(self,x):
        return self.conv(x)

# # in_ch, out_ch = 16,3
# in_ch, out_ch = 3,16
# model = UIB(in_ch, out_ch)
# x = torch.rand(128, in_ch, 64, 64)
# out = model(x)
# print(out.shape)
# # print(out)
