# @title pool tok mixer
# MetaFormer is Actually What You Need for Vision
# https://openaccess.thecvf.com/content/CVPR2022/papers/Yu_MetaFormer_Is_Actually_What_You_Need_for_Vision_CVPR_2022_paper.pdf

import torch.nn as nn
class Pooling(nn.Module):
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size//2, count_include_pad=False)
    def forward(self, x): # [b,c,h,w]
        return self.pool(x) - x # Subtraction of the input itself is added since the block already has a residual connection.

# model = Pooling()
# x = torch.randn(2, 3, 7,9)
# out = model(x)
# print(out.shape)
