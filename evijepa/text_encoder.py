import torch
import torch.nn as nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    def __init__(self, model_name, shared_dim=512, grad_ckpt=False):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        if grad_ckpt:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.config.use_cache = False
        self.hidden = self.model.config.hidden_size
        self.proj = nn.Linear(self.hidden, shared_dim)

    def forward(self, input_ids, attention_mask, return_tokens=False):
        h = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        out = self.proj(pooled)
        if return_tokens:
            return out, h
        return out
