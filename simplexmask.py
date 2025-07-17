# @title simplex
# !pip install -q opensimplex
import opensimplex
import numpy as np
import torch

def simplexmask2d(hw=(32,32), ctx_scale=(.85,1.), trg_scale=(.6,.8), B=64, chaos=[1,.5]):
    ix, iy = np.split(np.linspace(0, chaos[0], num=sum(hw)), [hw[0]])
    noise = opensimplex.noise3array(ix, iy, np.random.randint(1e10, size=B)) # [b,h,w]
    noise = torch.from_numpy(noise).flatten(1)
    trunc_normal = torch.fmod(torch.randn(2)*.3,1)/2 + .5
    ctx_mask_scale = torch.rand(1) * (ctx_scale[1] - ctx_scale[0]) + ctx_scale[0] # in (min_s, max_s) # all blocks same size
    trg_mask_scale = torch.rand(1) * (trg_scale[1] - trg_scale[0]) + trg_scale[0]
    # ctx_mask_scale = trunc_normal[0] * (ctx_scale[1] - ctx_scale[0]) + ctx_scale[0] # in (min_s, max_s) # all blocks same size
    # trg_mask_scale = trunc_normal[1] * (trg_scale[1] - trg_scale[0]) + trg_scale[0]
    seq = hw[0]*hw[1]
    ctx_len, trg_len = int(seq*ctx_mask_scale), int(seq*trg_mask_scale)
    val, trg_index = torch.topk(noise, trg_len, dim=1, sorted=False)
    # ctx_len = ctx_len - trg_len
    ctx_len = min(ctx_len, seq-trg_len)

    remove_mask = torch.ones((B,seq), dtype=bool) # [B, S]
    remove_mask.scatter_(1, trg_index, False).flatten()
    ind = torch.arange(seq).unsqueeze(0).repeat(B,1)[remove_mask].reshape(B, -1)

    ix, iy = np.split(np.linspace(0, chaos[1], num=sum(hw)), [hw[0]])
    noise = opensimplex.noise3array(ix, iy, np.random.randint(1e10, size=B)) # [b,h,w]
    noise = torch.from_numpy(noise).flatten(1)[remove_mask].reshape(B, -1)
    val, ctx_ind = torch.topk(noise, ctx_len, dim=1, sorted=False)
    ctx_index = ind[torch.arange(B).unsqueeze(-1), ctx_ind]
    return ctx_index, trg_index

b=16
# ctx_index, trg_index = simplexmask2d(hw=(32,32), ctx_scale=(.85,1), trg_scale=(.7,.8), B=b, chaos=[3,.5])
# ctx_index, trg_index = simplexmask2d(hw=(32,32), ctx_scale=(.85,1), trg_scale=(.7,.8), B=b, chaos=[3,1])
# ctx_index, trg_index = simplexmask2d(hw=(32,32), ctx_scale=(.85,1), trg_scale=(.2,.8), B=b, chaos=[3,1])
ctx_index, trg_index = simplexmask2d(hw=(32,32), ctx_scale=(.05,.2), trg_scale=(.7,.8), B=b, chaos=[3,1])
# ctx_index, trg_index = simplexmask2d(hw=(32,32), ctx_scale=(.85,1), trg_scale=(.5,.5), B=b, chaos=[.01,.5])


# import matplotlib.pyplot as plt
# def imshow(img):
#     npimg = img.numpy()
#     plt.rcParams["figure.figsize"] = (8,8)
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
# # imshow(mask)

# mask = torch.zeros(b ,32*32)
# # mask = torch.ones(b ,32*32)*.3
# mask[torch.arange(b).unsqueeze(-1), trg_index] = 1
# mask[torch.arange(b).unsqueeze(-1), ctx_index] = .5
# # print((mask==1).sum(1))
# # print((mask==.5).sum(1))
# mask = mask.reshape(b,1,32,32)
# import torchvision
# imshow(torchvision.utils.make_grid(mask, nrow=8))



# @title simplex
!pip install -q opensimplex
import opensimplex
import numpy as np
import torch

def simplexmask1d(seq=512, ctx_scale=(.85,1), trg_scale=(.6,.8), B=64, chaos=[1,.5]):
    i = np.linspace(0, chaos[0], num=seq) # 2-5
    noise = opensimplex.noise2array(i, np.random.randint(1e10, size=B)) # [B, seq]
    noise = torch.from_numpy(noise)
    # trunc_normal = torch.fmod(torch.randn(2)*.3,1)/2 + .5
    # print(trunc_normal)
    ctx_mask_scale = torch.rand(1) * (ctx_scale[1] - ctx_scale[0]) + ctx_scale[0] # in (min_s, max_s) # all blocks same size
    trg_mask_scale = torch.rand(1) * (trg_scale[1] - trg_scale[0]) + trg_scale[0]
    # ctx_mask_scale = trunc_normal[0] * (ctx_scale[1] - ctx_scale[0]) + ctx_scale[0] # in (min_s, max_s) # all blocks same size
    # trg_mask_scale = trunc_normal[1] * (trg_scale[1] - trg_scale[0]) + trg_scale[0]

    ctx_len, trg_len = int(seq*ctx_mask_scale), int(seq*trg_mask_scale)
    val, trg_index = torch.topk(noise, trg_len, dim=1, sorted=False)
    ctx_len = ctx_len - trg_len

    remove_mask = torch.ones((B,seq), dtype=bool) # [B, S]
    remove_mask.scatter_(1, trg_index, False).flatten()
    ind = torch.arange(seq).unsqueeze(0).repeat(B,1)[remove_mask].reshape(B, -1)

    i = np.linspace(0, chaos[1], num=seq) # 2-5
    noise = opensimplex.noise2array(i, np.random.randint(1e10, size=B)) # [B, seq]
    noise = torch.from_numpy(noise)[remove_mask].reshape(B, -1)
    val, ctx_ind = torch.topk(noise, ctx_len, dim=1, sorted=False)
    ctx_index = ind[torch.arange(B).unsqueeze(-1), ctx_ind]
    return ctx_index, trg_index

b=64
# ctx_index, trg_index = simplexmask1d(seq=200, ctx_scale=(.7,.8), trg_scale=(.4,.6), B=b, chaos=[3,.5])
ctx_index, trg_index = simplexmask1d(seq=200, ctx_scale=(.85,1), trg_scale=(.7,.8), B=b, chaos=[3,.5])
# ctx_index, trg_index = simplexmask1d(seq=200, ctx_scale=(.8,.9), trg_scale=(.7,.8), B=b, chaos=[1,.5])
mask = torch.zeros(b ,200)
mask[torch.arange(b).unsqueeze(-1), trg_index] = 1
mask[torch.arange(b).unsqueeze(-1), ctx_index] = .5
# mask = mask[None,...]
# mask = mask[:,None,None,:]#.repeat(1,3,1,1)
print(mask.shape)

# import matplotlib.pyplot as plt
# def imshow(img):
#     npimg = img.numpy()
#     plt.rcParams["figure.figsize"] = (8,4)
#     # plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.pcolormesh(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
# # imshow(mask[0])
# import torchvision
# imshow(torchvision.utils.make_grid(mask, nrow=1))

# print(index)
# print(index.shape)
# print(mask)
# print(mask.shape)
