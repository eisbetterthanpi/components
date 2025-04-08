# @title pixelshuffle_nd
import torch

def unshuffle(x, window_shape): # [b,c,h,w] -> [b, h/win1* w/win2, c, win1,win2]
    new_shape = list(x.shape[:2]) + [val for xx, win in zip(list(x.shape[2:]), window_shape) for val in [xx//win, win]] # [h,w]->[h/win1, win1, w/win2, win2]
    x = x.reshape(new_shape) # [b, c, h/win1, win1, w/win2, win2]
    # print('unsh',x.shape, window_shape, new_shape)
    L = len(new_shape)
    permute = ([0] + list(range(2, L - 1, 2)) + [1] + list(range(3, L, 2))) # [0,2,4,1,3,5] / [0,2,4,6,1,3,5,6]
    return x.permute(permute).flatten(1, L//2-1) # [b, h/win1* w/win2, c, win1,win2]

# @title shuffle

def shuffle(x, window_shape): # [b,win*win, c, h/win, w/win] -> [b,1,c,h,w]
    out_shape = [x.shape[0], -1, x.shape[2]] + [x*w for x,w in zip(x.shape[3:], window_shape)] # [b,1,c,h/win*wim, w/win*wim]
    # x = x.unflatten(0, (-1, *window_shape)) # [b,win,win, c, h/win, w/win]
    x = x.unflatten(1, (-1, *window_shape)) # [b,num_tok,win,win, c, h/win, w/win]
    D=x.dim()+1
    print('shf1', x.shape)
    # permute = [0,D//2+1] + [val for tup in zip(range(1+D//2, D+1), range(1, D//2)) for val in tup]
    permute = [0,1,D//2] + [val for tup in zip(range(1+D//2, D), range(2, D//2)) for val in tup]
    print('shf2', permute) # [0, 3, 4, 1, 5, 2]
    # x = x.permute(permute)
    print(out_shape)
    # x=x.reshape(*x.shape[:3], *out_shape)
    x = x.permute(permute).reshape(out_shape)
    return x

x = torch.rand(64, 4, 64, 8, 8)
out = shuffle(x, (2,2))
print(out.shape)

# # def shuffle(x, window_shape): # [b,win*win, c, h/win, w/win] -> [b,1,c,h,w]
# def shuffle(x, window_shape): # [b, h/win*w/win, c, win, win] -> [b,1,c,h,w]
#     out_shape = [x.shape[0], -1, x.shape[2]] + [x*w for x,w in zip(x.shape[3:], window_shape)] # [b,1,c,h/win*wim, w/win*wim]
#     x = x.unflatten(1, (-1, *window_shape)) # [b,num_tok,win,win, c, h/win, w/win]
#     D=x.dim()+1
#     permute = [0,1,D//2] + [val for tup in zip(range(1+D//2, D), range(2, D//2)) for val in tup]
#     x = x.permute(permute).reshape(out_shape)
#     return x


x = torch.rand(4,4)
print(x)
x=x.reshape(2,2,2,2) # [h/win1, win1, w/win2, win2]
print(x.permute(1,3,0,2).flatten(0,1))
print(x.permute(0,2,1,3).flatten(0,1))



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
