import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertFFN(nn.Module):
    def __init__(self, dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ExpertChoiceMoE(nn.Module):
    def __init__(self, dim, n_experts=8, out_dim=512, hidden=None, tau=0.5):
        super().__init__()
        hidden = hidden or dim * 2
        self.n_experts = n_experts
        self.tau = tau
        self.router = nn.Linear(dim, n_experts)
        self.experts = nn.ModuleList([ExpertFFN(dim, hidden, out_dim) for _ in range(n_experts)])

    def forward(self, x, mask):
        scores = self.router(x).float()
        scores = scores.masked_fill(~mask.bool().unsqueeze(-1), torch.finfo(scores.dtype).min)
        routing = (scores / self.tau).softmax(dim=1)
        weights = routing.to(x.dtype)
        evidence = [self.experts[e]((weights[:, :, e:e + 1] * x).sum(dim=1))
                    for e in range(self.n_experts)]
        return torch.stack(evidence, dim=1), routing


def route_diversity_loss(routing, mask):
    m = mask.bool().unsqueeze(-1).to(routing.dtype)
    a = routing * m
    a = a / (a.norm(dim=1, keepdim=True) + 1e-8)
    gram = torch.einsum("mne,mnf->mef", a, a)
    n_experts = routing.shape[-1]
    off = gram.sum(dim=(1, 2)) - torch.diagonal(gram, dim1=1, dim2=2).sum(-1)
    return (off / (n_experts * (n_experts - 1))).mean()


def evidence_diversity_loss(evidence):
    normed = F.normalize(evidence, dim=-1)
    gram = torch.einsum("med,mfd->mef", normed, normed) ** 2
    n_experts = evidence.shape[1]
    off = gram.sum(dim=(1, 2)) - torch.diagonal(gram, dim1=1, dim2=2).sum(-1)
    return (off / (n_experts * (n_experts - 1))).mean()
