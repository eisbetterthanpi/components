# @title MultiConv
import torch
from torch import nn
from torch.nn import functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MultiConv(nn.Module):
    def __init__(self, in_ch, out_ch=None, kernel_sizes=(3,7), **kwargs):
        super().__init__()
        out_ch = out_ch or in_ch
        # self.convs = nn.ModuleList([nn.Conv2d(in_ch, out_ch, kernel, padding=(kernel-stride)//2, **kwargs) for kernel in kernel_sizes])
        self.convs = nn.ModuleList([nn.Conv2d(in_ch, out_ch, kernel, padding=kernel//2, **kwargs) for kernel in kernel_sizes])
    def forward(self, x): return sum(conv(x) for conv in self.convs)
# https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338

# model = MultiConv(3,3,(3,7),2).to(device)
# model = MultiConv(4, 16, (3,7), groups=4, bias=False)
# print(sum(p.numel() for p in model.parameters() if p.requires_grad))
# x = torch.rand((2,3,16,16), device=device)
# out = model(x)
# print(out.shape)
