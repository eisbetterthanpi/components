# @title RankMe
# RankMe: Assessing the Downstream Performance of Pretrained Self-Supervised Representations by Their Rank jun 2023 https://arxiv.org/pdf/2210.02885
import torch

# https://github.com/Spidartist/IJEPA_endoscopy/blob/main/src/helper.py#L22
def RankMe(Z):
    """
    Calculate the RankMe score (the higher, the better).
    RankMe(Z) = exp(-sum_{k=1}^{min(N, K)} p_k * log(p_k)),
    where p_k = sigma_k (Z) / ||sigma_k (Z)||_1 + epsilon
    where sigma_k is the kth singular value of Z.
    where Z is the matrix of embeddings (N × K)
    """
    # compute the singular values of the embeddings
    # _u, s, _vh = torch.linalg.svd(Z, full_matrices=False)  # s.shape = (min(N, K),)
    # s = torch.linalg.svd(Z, full_matrices=False).S
    s = torch.linalg.svdvals(Z)
    p = s / torch.sum(s, axis=0) + 1e-7
    return torch.exp(-torch.sum(p * torch.log(p)))

Z = torch.randn(5, 3)
rankme = RankMe(Z)
print(rankme)
