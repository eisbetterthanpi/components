import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttn(nn.Module):
    def __init__(self, dim, n_heads, r=2):
        super().__init__()
        self.dim, self.heads, self.r = dim, n_heads, r
        d_head = dim//n_heads
        self.qkv = nn.Linear(dim, dim*3, bias=False)
        self.lin = nn.Conv2d(dim, dim, 1)
        self.rope = RoPE(d_head, seq_len=512, base=10000)
        # self.rope = RoPE2D(d_head, h=64, w=64, base=100)
        self.scale = d_head**-.5

    def forward(self, x): # [b,c,h,w]
        b,c,h,w = x.shape
        x = x.flatten(2).transpose(-2,-1)
        # x = F.pixel_unshuffle(x.transpose(0,1), self.r).flatten(2).permute(1,2,0) # [b,c,h,w] -> [c,b*r^2,h/r,w/r] -> [b,h/r*w/r,c]

        q,k,v = self.qkv(x).unflatten(-1, (self.heads,-1)).chunk(3, dim=-1) # [b, r^2, h/r*w/r, dim] # [b*r^2, h/r*w/r, n_heads, d_head]?
        q, k = self.rope(q), self.rope(k)

        q, k = q.softmax(dim=-1)*self.scale, k.softmax(dim=-2)
        context = k.transpose(-2,-1) @ v # [batch, n_heads, d_head, d_head]
        x = q @ context # [batch, n_heads, T/num_tok, d_head]

        x = x.transpose(-2,-1).reshape(b,c,h,w) # [batch, n_heads, T/num_tok, d_head] -> [batch, n_heads*d_head, T/num_tok] -> [b,c,h,w]
        # x = F.pixel_shuffle(x.flatten(2).permute(2,0,1).unflatten(-1, (h//self.r, w//self.r)), 2).transpose(0,1) # [b*r^2, h/r*w/r, n_heads, d_head] -> [d, b*r^2, h/r,w/r] -> [b,d,h,w]
        return self.lin(x)
