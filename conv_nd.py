import torch
import torch.nn as nn

def conv_nd(n, *args, **kwargs): return [nn.Identity, nn.Conv1d, nn.Conv2d, nn.Conv3d][n](*args, **kwargs)
# https://pytorch.org/docs/stable/nn.html#convolution-layers
# conv = conv_nd(2, 3,16,(3,3), 1,padding=3//2)
# # conv = nn.Conv2d(3,16,(3,3), 1,3//2) # Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# print(conv)
