# @title MultiHeadAttention
import torch
import torch.nn as nn
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads=None, d_head=8, cond_dim=None, dropout=0.): # .1
        super().__init__()
        self.d_model, self.n_heads, self.d_head = d_model, n_heads, d_model // n_heads
        self.cond_dim = cond_dim
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.kv = nn.Linear(cond_dim or d_model, 2*d_model, bias=False)
        self.lin = zero_module(nn.Linear(d_model, d_model))
        self.drop = nn.Dropout(dropout) # indp before q,k,v; after linout
        self.scale = self.d_head**-.5

    def forward(self, x, cond=None, mask=None): # [batch, T, d_model]=[batch, h*w, c], [batch, num_tok, cond_dim], [batch,T]
        if self.cond_dim==None: cond=x # is self attn
        q = self.q(x).unflatten(-1, (self.n_heads, self.d_head)).transpose(1, 2) # [batch, T, d_model] -> [batch, n_heads, T, d_head]
        # K = self.k(x).unflatten(-1, (self.n_heads, self.d_head)).transpose(1, 2)
        k, v = self.kv(cond).unflatten(-1, (self.n_heads, 2*self.d_head)).transpose(1, 2).chunk(2, dim=-1) # [batch, n_heads, T/num_tok, d_head]

        # # linear attention # Softmax(q) @ (Softmax(k).T @ v)
        # if mask != None:
        #     mask = mask[:, None, :, None] # [batch,T] -> [batch,1,T,1]
        #     k, v = k.masked_fill(mask, -torch.finfo(x.dtype).max), v.masked_fill(mask, -torch.finfo(x.dtype).max)
        # q, k = q.softmax(dim=-1)*self.scale, K.softmax(dim=-2)
        # context = k.transpose(-2,-1) @ v # [batch, n_heads, d_head, d_head]
        # out = q @ context # [batch, n_heads, T/num_tok, d_head]

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

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout=0):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.norm = nn.RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout=0)
        act = nn.ReLU()
        self.ff = nn.Sequential(
            nn.RMSNorm(d_model), nn.Linear(d_model, ff_dim), act,
            nn.RMSNorm(ff_dim), nn.Dropout(dropout), zero_module(nn.Linear(ff_dim, d_model))
        )

    def forward(self, x, mask=None):
        x = x + self.drop(self.attn(self.norm(x), mask=mask))
        x = x + self.drop(self.ff(x))
        return x
