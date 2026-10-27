import hashlib

import torch
import torch.nn.functional as F
import torch.distributed as dist


def dist_on():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def text_hashes(texts, device):
    values = [int(hashlib.md5(t.encode()).hexdigest()[:15], 16) for t in texts]
    return torch.tensor(values, dtype=torch.long, device=device)


class GatherLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        out = [torch.empty_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(out, x.contiguous())
        return torch.cat(out, dim=0)

    @staticmethod
    def backward(ctx, grad):
        dist.all_reduce(grad)
        world, rank = dist.get_world_size(), dist.get_rank()
        n = grad.shape[0] // world
        return grad[rank * n:(rank + 1) * n]


def _all_gather(x):
    out = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(out, x.contiguous())
    return torch.cat(out, dim=0)


def _pad(x, size):
    if x.shape[0] >= size:
        return x[:size]
    pad = x.new_zeros((size - x.shape[0],) + tuple(x.shape[1:]))
    return torch.cat([x, pad], dim=0)


def gather_all(pred, tgt, normal, thash, batch_size):
    local = pred.shape[0]
    if not dist_on():
        valid = torch.ones(local, dtype=torch.bool, device=pred.device)
        return 0, pred, tgt, normal.bool(), thash, valid, local, local
    rank = dist.get_rank()
    valid = torch.zeros(batch_size, dtype=torch.bool, device=pred.device)
    valid[:local] = True
    pred_all = GatherLayer.apply(_pad(pred, batch_size))
    tgt_all = GatherLayer.apply(_pad(tgt, batch_size))
    normal_all = _all_gather(_pad(normal.long(), batch_size)).bool()
    thash_all = _all_gather(_pad(thash, batch_size)).long()
    valid_all = _all_gather(valid.long()).bool()
    return rank, pred_all, tgt_all, normal_all, thash_all, valid_all, local, batch_size


def soft_targets(normal_anchor, normal_all, hash_anchor, hash_all, valid_all, offset, n_anchor):
    device = normal_all.device
    n_all = normal_all.shape[0]
    rows = torch.arange(n_anchor, device=device)
    cols = offset + rows
    positive = torch.zeros(n_anchor, n_all, device=device)
    positive[rows, cols] = 1.0
    both_normal = normal_anchor.view(-1, 1) & normal_all.view(1, -1)
    same_text = hash_anchor.view(-1, 1) == hash_all.view(1, -1)
    semantic = ((both_normal | same_text) & valid_all.view(1, -1)).float()
    semantic[rows, cols] = 0.0
    targets = positive + semantic
    return targets / targets.sum(1, keepdim=True)


def masked_nce(sim, targets, valid_all):
    sim = sim.masked_fill(~valid_all[None], float("-inf"))
    logp = F.log_softmax(sim, dim=1).masked_fill(~valid_all[None], 0.0)
    return -(logp * targets).sum(1).mean()


def organ_infonce(pred, tgt, normal, thash, batch_size, temperature=0.07):
    rank, pred_all, tgt_all, normal_all, thash_all, valid_all, n_anchor, size = \
        gather_all(pred, tgt, normal, thash, batch_size)
    if n_anchor < 2 and not dist_on():
        return None
    p = F.normalize(pred_all.float(), dim=-1)
    t = F.normalize(tgt_all.float(), dim=-1)
    offset = rank * size
    anchor_normal = normal_all[offset:offset + n_anchor]
    anchor_hash = thash_all[offset:offset + n_anchor]
    with torch.no_grad():
        targets = soft_targets(anchor_normal, normal_all, anchor_hash, thash_all,
                               valid_all, offset, n_anchor)
        if targets.unique().numel() == 1:
            return None
    sim_i2t = p[offset:offset + n_anchor] @ t.t() / temperature
    sim_t2i = t[offset:offset + n_anchor] @ p.t() / temperature
    return (masked_nce(sim_i2t, targets, valid_all) + masked_nce(sim_t2i, targets, valid_all)) / 2


def sinkhorn(cost, n_iter=5, eps=0.05):
    log_p = cost / eps
    for _ in range(n_iter):
        log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
        log_p = log_p - torch.logsumexp(log_p, dim=-2, keepdim=True)
    return log_p.exp()


def evidence_infonce(vision, text, normal, thash, temperature=0.07, n_iter=5, eps=0.05):
    n, n_experts, _ = vision.shape
    if n < 2:
        return None
    v = F.normalize(vision.float(), dim=-1)
    t = F.normalize(text.float(), dim=-1)
    cost = torch.einsum("med,nfd->mnef", v, t)
    plan = sinkhorn(cost, n_iter, eps)
    sim = (plan * cost).sum(dim=(-1, -2)) / n_experts / temperature
    with torch.no_grad():
        positive = torch.eye(n, device=vision.device)
        both_normal = normal.view(n, 1) & normal.view(1, n)
        same_text = thash.view(n, 1) == thash.view(1, n)
        semantic = (both_normal | same_text).float()
        semantic.fill_diagonal_(0)
        targets = positive + semantic
        targets = targets / targets.sum(1, keepdim=True)
        if targets.unique().numel() == 1:
            return None
    loss_i2t = -(F.log_softmax(sim, dim=1) * targets).sum(1).mean()
    loss_t2i = -(F.log_softmax(sim.t(), dim=1) * targets).sum(1).mean()
    return (loss_i2t + loss_t2i) / 2


class AsymmetricLoss(torch.nn.Module):
    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05):
        super().__init__()
        self.gamma_neg, self.gamma_pos, self.clip = gamma_neg, gamma_pos, clip

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        p_neg = (p - self.clip).clamp(min=0) if self.clip > 0 else p
        loss_pos = targets * torch.log(p.clamp(min=1e-8)) * (1 - p) ** self.gamma_pos
        loss_neg = (1 - targets) * torch.log((1 - p_neg).clamp(min=1e-8)) * p_neg ** self.gamma_neg
        return -(loss_pos + loss_neg).sum(dim=-1).mean()
