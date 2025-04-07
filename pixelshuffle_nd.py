# @title pixelshuffle_nd
import torch

def unshuffle(x, window_shape): # [b,c,h,w] -> [b*win*win, c, h/win, w/win]
    new_shape = list(x.shape[:2]) + [val for xx,win in zip(list(x.shape[2:]), window_shape) for val in [xx//win,win]]
    x = x.reshape(new_shape)
    L = len(new_shape)
    permute = ([0] + list(range(3, L, 2)) + [1] + list(range(2, L - 1, 2))) # [0,3,5,1,2,4] / [0,3,5,7,1,2,4,6]
    x = x.permute(permute).flatten(end_dim=L//2-1) # [b*win*win, c, h/win, w/win]
    return x

def shuffle(x, window_shape): # [b*win*win, c, h/win, w/win] -> [b,c,h,w]
    out_shape = [x*w for x,w in zip(x.shape[2:], window_shape)] # [h/win*wim, w/win*wim]
    x = x.unflatten(0, (-1, *window_shape)) # [b,win,win, c, h/win, w/win]
    D=x.dim()
    permute = [0,D//2] + [val for tup in zip(range(1+D//2, D+1), range(1, D//2)) for val in tup]
    x = x.permute(permute).reshape(*x.shape[:2], *out_shape)
    return x
