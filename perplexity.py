# @title perplexity
import torch
import torch.nn.functional as F
# https://www.comet.com/site/blog/perplexity-for-llm-evaluation/

def Perplexity(logits, target): # [b,t,vocab_size], [b,t]
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1) # [b,t]
    perplexity = nll.mean().exp()
    return perplexity

# logits = torch.randn(2, 4, 10)
# target = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])

# perplexity = Perplexity(logits, target)
# # perplexity = Perplexity(y_[:,;-1], y[:,1:])
# print(f'Perplexity: {perplexity}')
