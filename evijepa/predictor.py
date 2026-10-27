import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM


class OrganPredictor(nn.Module):
    def __init__(self, llama_name, n_keep_layers=8, vis_dim=1024, shared_dim=512,
                 grad_ckpt=False, n_organs=4):
        super().__init__()
        full = AutoModelForCausalLM.from_pretrained(llama_name, attn_implementation="sdpa")
        base = full.model
        self.hidden = base.config.hidden_size
        self.layers = nn.ModuleList(list(base.layers[-n_keep_layers:]))
        self.norm = base.norm
        self.rotary_emb = getattr(base, "rotary_emb", None)
        self.vis_proj = nn.Linear(vis_dim, self.hidden)
        self.out_proj = nn.Linear(self.hidden, shared_dim)
        self.pool_query = nn.Parameter(torch.randn(n_organs, self.hidden) * 0.02)
        self.grad_ckpt = grad_ckpt

    @property
    def vis_w(self):
        return self.vis_proj.weight

    def _run_layers(self, h, mask):
        B, N, _ = h.shape
        attn = mask[:, None, None, :].to(h.dtype)
        attn = (1.0 - attn) * torch.finfo(h.dtype).min
        pos_ids = torch.arange(N, device=h.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.rotary_emb(h, pos_ids) if self.rotary_emb is not None else None
        for layer in self.layers:
            if self.grad_ckpt and self.training:
                h = checkpoint(lambda x, l=layer: l(x, attention_mask=attn, position_ids=pos_ids,
                                                    position_embeddings=pos_emb)[0],
                               h, use_reentrant=False)
            else:
                h = layer(h, attention_mask=attn, position_ids=pos_ids,
                          position_embeddings=pos_emb)[0]
        return self.norm(h)

    def _pool(self, h, mask, organ_idx):
        q = self.pool_query[organ_idx:organ_idx + 1]
        score = (h @ q.to(h.dtype).t()).squeeze(-1)
        score = score.float().masked_fill(~mask.bool(), torch.finfo(torch.float32).min)
        attn = score.softmax(dim=1).to(h.dtype)
        return (attn.unsqueeze(-1) * h).sum(1)

    def forward(self, vis_tokens, vis_mask, return_tokens=False, organ_idx=0):
        h = self.vis_proj(vis_tokens)
        h = self._run_layers(h, vis_mask)
        pooled = self.out_proj(self._pool(h, vis_mask, organ_idx))
        if return_tokens:
            return pooled, h
        return pooled
