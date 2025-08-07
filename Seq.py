import inspect
class Seq(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)
        for layer in self:
            params = inspect.signature(layer.forward).parameters.keys()
            layer._fwdparams = ','.join(params)

    def forward(self, x, cond=None, masks=None):
        arg_map = {'cond':cond, 'masks':masks}
        for layer in self:
            args = [x]
            # if 'cond' in layer._fwdparams: args.append(cond)
            # if 'masks' in layer._fwdparams: args.append(masks)
            args.extend(arg_map[p] for p in arg_map if p in layer._fwdparams)
            # print(layer._fwdparams, args)
            x = layer(*args)
        return x


