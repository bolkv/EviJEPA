import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_backbone(ckpt_path, arch, grad_ckpt):
    repo = os.environ.get("VJEPA_REPO")
    if repo and repo not in sys.path:
        sys.path.insert(0, repo)
    from src.hub.backbones import (
        vjepa2_1_vit_base_384,
        vjepa2_1_vit_large_384,
        _clean_backbone_key,
    )

    builder = {"base": vjepa2_1_vit_base_384, "large": vjepa2_1_vit_large_384}[arch]
    encoder, _ = builder(pretrained=False, use_activation_checkpointing=grad_ckpt)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    encoder.load_state_dict(_clean_backbone_key(state["ema_encoder"]), strict=False)
    return encoder


class VJEPAOrganEncoder(nn.Module):
    def __init__(self, ckpt_path, arch="large", n_organs=4, patch=16, tubelet=2,
                 lora=True, lora_r=16, lora_alpha=32, lora_dropout=0.05,
                 grad_ckpt=False, z_unshuffle=2, patch_embed_full=True):
        super().__init__()
        encoder = _load_backbone(ckpt_path, arch, grad_ckpt)
        self.hidden = encoder.embed_dim
        self.z_unshuffle = z_unshuffle
        self.patch_embed_full = patch_embed_full
        self.patch, self.tubelet, self.n_organs = patch, tubelet, n_organs
        self.intensity_aug = None

        if z_unshuffle > 1:
            old = encoder.patch_embed.proj
            new = nn.Conv3d(old.in_channels * z_unshuffle, old.out_channels,
                            kernel_size=old.kernel_size, stride=old.stride)
            with torch.no_grad():
                new.weight.copy_(old.weight.repeat_interleave(z_unshuffle, dim=1) / z_unshuffle)
                new.bias.copy_(old.bias)
            encoder.patch_embed.proj = new

        if lora:
            encoder.requires_grad_(False)
            target = "attn.proj" if patch_embed_full else "proj"
            encoder = get_peft_model(encoder, LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                target_modules=["qkv", target], bias="none"))
            if patch_embed_full:
                for name, param in encoder.named_parameters():
                    if "patch_embed" in name and "lora" not in name:
                        param.requires_grad_(True)
        else:
            encoder.requires_grad_(True)

        self.model = encoder
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def _augment_intensity(self, vid):
        cfg = self.intensity_aug
        batch = vid.shape[0]
        device = vid.device

        def gate(p):
            return (torch.rand(batch, device=device) < p).view(-1, 1, 1, 1, 1)

        def value(low, high):
            return (low + (high - low) * torch.rand(batch, device=device)).view(-1, 1, 1, 1, 1)

        if cfg.get("brightness_p", 0) > 0:
            m = gate(cfg["brightness_p"])
            vid = torch.where(m, (vid * value(*cfg.get("brightness", [0.9, 1.1]))).clamp(0, 1), vid)
        if cfg.get("contrast_p", 0) > 0:
            m = gate(cfg["contrast_p"])
            mean = vid.mean(dim=(1, 3, 4), keepdim=True)
            vid = torch.where(m, ((vid - mean) * value(*cfg.get("contrast", [0.9, 1.1])) + mean).clamp(0, 1), vid)
        if cfg.get("gamma_p", 0) > 0:
            m = gate(cfg["gamma_p"])
            vid = torch.where(m, vid.clamp(1e-5, 1).pow(value(*cfg.get("gamma", [0.85, 1.2]))), vid)
        return vid

    def organ_flags(self, mask, T, H, W, L, device):
        zk = self.tubelet * self.z_unshuffle
        gd, gh, gw = T // zk, H // self.patch, W // self.patch
        assert gd * gh * gw == L
        flags = torch.zeros(mask.shape[0], self.n_organs, L, dtype=torch.bool, device=device)
        for o in range(self.n_organs):
            binary = (mask == o + 1).float().unsqueeze(1)
            pooled = F.max_pool3d(binary, kernel_size=(zk, self.patch, self.patch),
                                  stride=(zk, self.patch, self.patch))
            flags[:, o] = pooled.flatten(1) > 0
        return flags

    def forward(self, video, mask=None):
        B, T, H, W = video.shape
        video = video.float()
        hu = video * (350.0 + 1150.0) - 1150.0
        lung = ((hu + 1000.0) / 1400.0).clamp(0, 1)
        soft = ((hu + 150.0) / 400.0).clamp(0, 1)
        wide = video
        vid = torch.stack([lung, soft, wide], dim=2)
        if self.training and self.intensity_aug:
            vid = self._augment_intensity(vid)
        vid = (vid - self.mean.view(1, 1, 3, 1, 1).to(vid.dtype)) / self.std.view(1, 1, 3, 1, 1).to(vid.dtype)
        vid = vid.permute(0, 2, 1, 3, 4)

        zu = self.z_unshuffle
        if zu > 1:
            Bv, Cv, Tv, Hv, Wv = vid.shape
            vid = vid.view(Bv, Cv, Tv // zu, zu, Hv, Wv).permute(0, 1, 3, 2, 4, 5)
            vid = vid.reshape(Bv, Cv * zu, Tv // zu, Hv, Wv)

        tokens = self.model(vid)
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[-1]

        flags = None
        if mask is not None:
            flags = self.organ_flags(mask, T, H, W, tokens.shape[1], video.device)
        return tokens, flags
