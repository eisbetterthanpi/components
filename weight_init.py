import math
import torch
import torch.nn as nn

# https://github.com/facebookresearch/blt/blob/main/bytelatent/model/local_models.py#L136
# torch.nn.init.normal_(self.lin0.weight, std=.02)
torch.nn.init.normal_(self.lin0.weight, std=1/(math.sqrt(d_model)+math.sqrt(ff_dim)))
nn.init.trunc_normal_(self.tok_emb.weight, mean=0, std=emb_std, a=-3*emb_std, b=3*emb_std) # https://github.com/facebookresearch/blt/blob/main/bytelatent/model/local_models.py#L136


def StableInit(m): # https://openreview.net/pdf?id=lkRjnNW0gb
    if isinstance(m, nn.Linear):
        # W ~ N(0, ( 1/ (sqrt(n_in) + sqrt(n_out)) )^2 )
        # want std = 1/ (sqrt(n_in) + sqrt(n_out))
        # n_in, n_out = module.weight.shape[0], module.weight.shape[1]
        n_in, n_out = m.weight.shape
        nn.init.normal_(m.weight, std=1/(math.sqrt(n_in)+math.sqrt(n_out)))
        if m.bias is not None:
            nn.init.zeros_(m.bias)


        self.embed.apply(self.init_conv)
        self.embed.apply(self.init_weights)

    def init_conv(self, m):
        if isinstance(m, nn.Conv1d):
            # nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                # bound = 1 / math.sqrt(m.in_channels * m.kernel_size * m.kernel_size)
                # nn.init.uniform_(m.bias, -bound, bound)
                nn.init.zeros_(m.bias)

    def init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.zeros_(m.bias)



