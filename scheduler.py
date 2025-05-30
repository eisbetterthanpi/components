
# @title scheduler
from torch.optim.lr_scheduler import LambdaLR
import math
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

# total_steps=100
# base_lr, max_lr = 3e-5, 3e-4

# import torch
# model=torch.nn.Linear(2,3)
# optim = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.999))

# scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps=20 , num_training_steps=total_steps) # https://docs.pytorch.org/torchtune/0.2/generated/torchtune.modules.get_cosine_schedule_with_warmup.html
# # scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=max_lr, total_steps=total_steps, pct_start=0.45, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=div_factor, final_div_factor=100.0, three_phase=True,)

# lr_lst=[]
# import matplotlib.pyplot as plt
# for t in range(total_steps):
#     lr=optim.param_groups[0]["lr"]
#     lr_lst.append(lr)
#     scheduler.step()
# plt.plot(lr_lst)


