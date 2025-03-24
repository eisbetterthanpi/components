# # # https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
# # # https://github.com/DingXiaoH/RepVGG/blob/main/repvggplus.py#L28
# class FeedForward(nn.Module):
#     def __init__(self, in_ch, out_ch=None, ff_mult=1):
#         super().__init__()
#         # d_model = dim*ff_mult
#         out_ch = out_ch or in_ch
#         self.dwconv = nn.ModuleList([
#             nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, 1, 3//2), nn.BatchNorm2d(out_ch)),
#             nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, 1, 1//2), nn.BatchNorm2d(out_ch)),
#         ])
#         self.project_out = nn.Sequential(nn.GELU(), SEBlock(out_ch, out_ch//4)) # act, SEblock

#     def forward(self, x):
#         x = x + sum([conv(x) for conv in self.dwconv])
#         x = self.project_out(x)
#         return x

# # me
class FeedForward(nn.Module):
    def __init__(self, d_model, ff_mult=1):
        super().__init__()
        # d_model = dim*ff_mult
        self.project_in = nn.Sequential(nn.BatchNorm2d(d_model), nn.GELU())
        self.dwconv = nn.ModuleList([
            nn.Conv2d(d_model, d_model, 3, 1, 3//2),
            nn.Conv2d(d_model, d_model, 1, 1, 1//2),
        ])
        # self.project_out = nn.Sequential(nn.GELU(), nn.Conv2d(d_model, dim, kernel_size=1)) # act, SEblock

    def forward(self, x):
        h = self.project_in(x) # no
        x = x + sum([conv(h) for conv in self.dwconv])
        # x = self.project_out(out)
        return x
