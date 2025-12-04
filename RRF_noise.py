# @title Random Fourier Features noise
import math
import torch

# Random Fourier Features
# kernel
# finite random Fourier expansion: RFF(x) = ∑_k] a_k * cos(ω_k ⋅ x + 𝜙_k)
# unbiased, real valued estimate of k(x-y) by sampling w,phi and computing sqrt2*cos(wTx+b)

def rff_noise(x, out_dim, n_freqs=64, scale=1): # [...,n_Dim]
    space, in_dim, device = x.shape[:-1], x.shape[-1], x.device
    x = x.flatten(0,-2)
    omega = torch.randn(out_dim, n_freqs, in_dim, device=device) * scale
    phi = torch.empty(out_dim, n_freqs, device=device).uniform_(0,2*math.pi)
    a = torch.randn(out_dim, n_freqs, device=device) / math.sqrt(n_freqs)
    # proj = omega @ x.T # [freq,in]@[N,in].T->[...,freq]
    # proj = x[None,] @ omega.T # [N,in]@[out,freq,in].T->[N,out,freq]
    # proj = torch.einsum("...d,ofd->...of", x, omega) # [...,in].[out,freq,in]->[...,out,freq]
    proj = torch.einsum("ofd,...d->...of", omega, x) # [out,freq,in].[...,in]->[...,out,freq]
    y = torch.cos(proj + phi) * a # [...,out,freq]
    # y = (2/n_freqs)**.5*torch.cos(proj + phi) * a # [...,out,freq]
    return y.sum(dim=-1).unflatten(0,space) # [...,out]
    # return torch.einsum("of,...of->...o", a, y)
# noise = noise.reshape(t,h,w,b)

# Random Fourier Features: ∑_k] a_k * cos(ω_k ⋅ x + 𝜙_k)
def rff_noise(x, out_dim, n_freqs=64, scale=1): # [...,n_Dim]
    space, in_dim, device = x.shape[:-1], x.shape[-1], x.device
    x = x.flatten(0,-2) # [N,in]
    w = torch.randn(out_dim, n_freqs, in_dim, device=device) * scale
    phi = torch.empty(out_dim, n_freqs, device=device).uniform_(0,2*math.pi)
    a = torch.randn(out_dim, n_freqs, device=device) / math.sqrt(n_freqs)
    y = torch.cos(torch.einsum("ofd,...d->...of", w, x) + phi) # [N,out,freq]
    return torch.einsum("of,...of->...o", a, y).unflatten(0,space) # [...,out]

# 4 octaves of RFF noise
# 32 Fourier frequencies per octave
# Frequencies double each octave (lacunarity)
# Amplitude halves each octave (gain)
def rff_fbm(x, out_dim, n_freqs=8, n_octave=4, lacunarity=2, gain=.5):
    tt_noise = 0
    amp, scale = 1, 1
    for _ in range(n_octave):
        tt_noise += amp * rff_noise(x, out_dim, n_freqs=n_freqs, scale=scale)
        amp, scale = amp*gain, scale*lacunarity
    return tt_noise#.reshape(*x.shape,out_dim)

b,t,h,w = 16,1,64,64
# b,t,h,w = 16,240,16,16
# b,t,h,w = 64,1,200,16
s = 5
x = torch.stack(torch.meshgrid(
    torch.linspace(0,s,t),
    torch.linspace(0,s,h),
    torch.linspace(0,s,w),
    indexing="ij"), dim=-1) # [b,t,h,w,4] / [b,h,w,3]
import time
t0 = time.time()
noise = rff_noise(x, b, n_freqs=16) # [t*h*w*in]->[t*h*w*b] / [h*w*b]
print(time.time()-t0)
# noise = rff_fbm(x, b, n_freqs=128) # [t*h*w*in]->[t*h*w*b] / [h*w*b]

noise = noise.permute(3,0,1,2)

import numpy as np
import matplotlib.pyplot as plt
def imshow(img):
    npimg = img.numpy()
    plt.rcParams["figure.figsize"] = (8,8)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.imshow(npimg)
    plt.show()
# imshow(noise[0,0])
import torchvision
# imshow(torchvision.utils.make_grid((noise[:,0].reshape(b,1,h,w)>0).float(), nrow=8))
imshow(torchvision.utils.make_grid(noise[:,0].reshape(b,1,h,w), nrow=8))
# imshow(torchvision.utils.make_grid(noise[0].reshape(t,1,h,w), nrow=8))

