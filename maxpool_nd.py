import torch
import torch.nn as nn

def maxpool_nd(n, *args, **kwargs): return [nn.Identity, nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d][n](*args, **kwargs)
# https://pytorch.org/docs/stable/nn.html#pooling-layers
# pool = maxpool_nd(3, (3, 2, 2), stride=(2, 1, 2))
# print(pool)
