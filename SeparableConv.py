# @title SeparableConv
# Xception: Deep Learning with Depthwise Separable Convolutions https://arxiv.org/pdf/1610.02357
class SeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch=None):
        super().__init__()
        out_ch = out_ch or in_ch
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 3//2, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self,x):
        x = self.conv(x)
        x = self.pointwise(x)
        return x

class SeparableConv1d(nn.Module):
    def __init__(self, in_ch, out_ch=None):
        super().__init__()
        out_ch = out_ch or in_ch
        self.conv = nn.Convd(in_ch, out_ch, 3, 1, 3//2, groups=in_ch, bias=False)
        self.pointwise = nn.Convd(in_ch, out_ch, 1, bias=False)

    def forward(self,x):
        x = self.conv(x)
        x = self.pointwise(x)
        return x

# #  'all Convolution and SeparableConvolution layers are followed by batch normalization'?
# in_ch, out_ch = 3, 3
# model = SeparableConv2d(in_ch, out_ch)
# x = torch.randn(2, in_ch, 7,9)
# out = model(x)
# print(out.shape)
