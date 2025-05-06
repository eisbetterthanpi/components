
def soft_clamp_relu(x, lower=None, upper=None):
    if lower != None: x = lower+F.relu(x-lower)
    if upper != None: x = upper-F.relu(upper-x)
    return x
