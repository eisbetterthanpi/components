# @title Attention
import torch
import torch.nn as nn
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

class MultiHeadAttention(nn.Module):
    # def __init__(self, d_model, n_heads=None, d_head=8, cond_dim=None, dropout=0.): # .1
    def __init__(self, query_dim, cond_dim=None, n_heads=8, d_head=64, drop=0):
        super().__init__()
        # self.d_model, self.n_heads, self.d_head = d_model, n_heads, d_model // n_heads
        d_model = d_head * n_heads
        self.d_head, self.n_heads = d_head, n_heads
        self.cond_dim = cond_dim
        self.q = nn.Linear(query_dim, d_model, bias=False)
        self.kv = nn.Linear(cond_dim or d_model, 2*d_model, bias=False)
        self.lin = zero_module(nn.Linear(d_model, d_model))
        self.drop = nn.Dropout(dropout) # indp before q,k,v; after linout
        self.scale = self.d_head**-.5
        # torch.nn.init.normal_(self.q.weight, std=.02)
        # torch.nn.init.normal_(self.kv.weight, std=.02)
        torch.nn.init.normal_(self.q.weight, std=1/(math.sqrt(query_dim)+math.sqrt(d_model)))
        torch.nn.init.normal_(self.kv.weight, std=1/(math.sqrt(cond_dim or query_dim)+math.sqrt(d_model)))

    def forward(self, x, cond=None, mask=None): # [batch, T, d_model]=[batch, h*w, c], [batch, num_tok, cond_dim], [batch,T]
        if self.cond_dim==None: cond=x # is self attn
        q = self.q(x).unflatten(-1, (self.n_heads, self.d_head)).transpose(1, 2) # [batch, T, d_model] -> [batch, n_heads, T, d_head]
        # K = self.k(x).unflatten(-1, (self.n_heads, self.d_head)).transpose(1, 2)
        k, v = self.kv(cond).unflatten(-1, (self.n_heads, 2*self.d_head)).transpose(1, 2).chunk(2, dim=-1) # [batch, n_heads, T/num_tok, d_head]

        # # (quadratic) attention # Softmax(q @ k.T) @ v
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.unsqueeze(1) if mask != None else None, dropout_p=0) # mask: [batch,len_q, len_v]
        # out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0) # mask: [batch,len_q, len_v]
        # attn = q @ k.transpose(-2,-1) * self.scale # [batch, n_heads, T] # [batch, n_heads, T, T/num_tok]
        # # if mask != None: attn = attn.masked_fill(mask[:, None, :, None], -torch.finfo(attn.dtype).max) # [batch,T]->[batch,1,T,1]
        # if mask != None: attn = attn.masked_fill(mask.unsqueeze(1), -torch.finfo(attn.dtype).max) # [b,t,t]->[b,1,t,t]
        # attention = torch.softmax(attn, dim=-1)
        # out = self.drop(attention) @ v # [batch, n_heads, T, d_head]

        out = out.transpose(1,2).flatten(2)
        return self.drop(self.lin(out)) # [batch, T, d_model]


class SwiGLU(nn.Module): # https://arxiv.org/pdf/2002.05202
    def __init__(self, d_model, ff_dim): # d_model * 3*ff_dim params
        super().__init__()
        self.lin0 = nn.Linear(d_model, 2*ff_dim, bias=False)
        self.lin1 = zero_module(nn.Linear(ff_dim, d_model, bias=False))
        # torch.nn.init.normal_(self.lin0.weight, std=.02)
        torch.nn.init.normal_(self.lin0.weight, std=1/(math.sqrt(d_model)+math.sqrt(ff_dim)))

    def forward(self, x): # [b,t,d]
        x0, x1 = self.lin0(x).chunk(2, dim=-1)
        return self.lin1(x0*F.silu(x1))


class AttentionBlock(nn.Module):
    # def __init__(self, d_model, cond_dim=None, d_head, ff_dim=None, dropout=0.):
    def __init__(self, d_model, n_heads ,cond_dim=None, ff_dim=None, drop=0):
        super().__init__()
        self.d_model = d_model
        self.cond_dim = cond_dim
        self.norm1 = nn.RMSNorm(d_model) # LayerNorm RMSNorm
        if cond_dim!=None: self.norm2 = nn.RMSNorm(cond_dim)
        self.drop = nn.Dropout(drop)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.attn = Attention(d_model, cond_dim, n_heads=n_heads, d_head=d_model//n_heads)
        act = nn.ReLU()
        if ff_dim==None: ff_dim=d_model*4
        self.ff = nn.Sequential(
            nn.RMSNorm(d_model), nn.Linear(d_model, ff_dim), act,
            nn.RMSNorm(ff_dim), zero_module(nn.Linear(ff_dim, d_model))
            # nn.RMSNorm(d_model), act, nn.Linear(d_model, ff_dim),
            # nn.RMSNorm(ff_dim), act, zero_module(nn.Linear(ff_dim, d_model))
        )
        # self.ff = SwiGLU(d_model, ff_dim)
        # torch.nn.init.normal_(self.ff[1].weight, std=.02)
        torch.nn.init.normal_(self.ff[1].weight, std=1/(math.sqrt(d_model)+math.sqrt(ff_dim)))

    def forward(self, x, cond=None, mask=None): # [b,c,h,w], [batch, num_tok, cond_dim], [batch,T]
        # print('attblk fwd', x.shape, cond.shape if cond!=None else None, mask.shape if mask!=None else None)
        if self.cond_dim==None: x = x + self.attn(self.norm1(x), mask=mask)
        else: x = x + self.attn(self.norm1(x), self.norm2(cond), mask) # maybe no res for decoder
        x = x + self.ff(x) # maybe no ff for decoder?
        return x

import inspect
class Seq(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)
        for layer in self:
            params = inspect.signature(layer.forward).parameters.keys()
            layer._fwdparams = ','.join(params)

    def forward(self, x, cond=None, mask=None):
        for layer in self:
            args = [x]
            if 'cond' in layer._fwdparams: args.append(cond)
            if 'mask' in layer._fwdparams: args.append(mask)
            x = layer(*args)
        return x

# b,t,d = 2,5,16
# x = torch.rand(b,t,d)
# mask = torch.rand(b,t,t)>0
# model = AttentionBlock(d_model=d, n_heads=4, ff_dim=16)
# model = nn.Sequential(*[AttentionBlock(d_model=d, n_heads=4, ff_dim=16) for _ in range(2)])
# out =  model(x, mask)
# # out =  model(x)
# print(out.shape)
