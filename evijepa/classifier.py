import torch
import torch.nn as nn

from .vision_encoder import VJEPAOrganEncoder

LABELS = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly", "Pericardial effusion",
    "Coronary artery wall calcification", "Hiatal hernia", "Lymphadenopathy", "Emphysema",
    "Atelectasis", "Lung nodule", "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion",
    "Mosaic attenuation pattern", "Peribronchial thickening", "Consolidation", "Bronchiectasis",
    "Interlobular septal thickening",
]


class LabelAttnHead(nn.Module):
    def __init__(self, dim, n_labels=18, n_heads=8, ffn_mult=2, p_drop=0.1):
        super().__init__()
        assert dim % n_heads == 0
        self.n_labels, self.n_heads, self.head_dim = n_labels, n_heads, dim // n_heads
        self.query = nn.Parameter(torch.randn(n_labels, dim) * dim ** -0.5)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.norm_tokens = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)
        self.drop = nn.Dropout(p_drop)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult), nn.GELU(),
            nn.Dropout(p_drop), nn.Linear(dim * ffn_mult, dim))
        self.cls_w = nn.Parameter(torch.randn(n_labels, dim) * dim ** -0.5)
        self.cls_b = nn.Parameter(torch.zeros(n_labels))

    def forward(self, tokens, return_attn=False):
        B, L, D = tokens.shape
        H, dh = self.n_heads, self.head_dim
        t = self.norm_tokens(tokens)
        k = self.k_proj(t).view(B, L, H, dh).transpose(1, 2)
        v = self.v_proj(t).view(B, L, H, dh).transpose(1, 2)
        q = self.q_proj(self.query).view(1, self.n_labels, H, dh).permute(0, 2, 1, 3).expand(B, -1, -1, -1)
        attn = ((q @ k.transpose(-2, -1)) * dh ** -0.5).softmax(-1)
        h = (self.drop(attn) @ v).transpose(1, 2).reshape(B, self.n_labels, D)
        h = self.norm_out(h)
        h = h + self.ffn(h)
        logits = (h * self.cls_w).sum(-1) + self.cls_b
        if return_attn:
            return logits, attn.mean(1)
        return logits


class CTLabelClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        model_cfg = cfg["model"]
        self.vision = VJEPAOrganEncoder(
            model_cfg["vjepa21_ckpt"],
            arch=model_cfg.get("vision_arch", "vjepa21_large").split("_")[1],
            n_organs=4,
            lora=model_cfg.get("vision_lora", True),
            lora_r=model_cfg.get("lora_r", 16),
            lora_alpha=model_cfg.get("lora_alpha", 32),
            lora_dropout=model_cfg.get("lora_dropout", 0.05),
            grad_ckpt=model_cfg.get("gradient_checkpointing", False),
            z_unshuffle=model_cfg.get("z_unshuffle", 2),
            patch_embed_full=model_cfg.get("patch_embed_full", True))

        augment = (cfg.get("data") or {}).get("augment") or {}
        intensity_keys = ("brightness_p", "brightness", "contrast_p", "contrast",
                          "gamma_p", "gamma", "gamma_invert")
        intensity = {k: augment[k] for k in intensity_keys if k in augment}
        self.vision.intensity_aug = intensity or None

        self.head = LabelAttnHead(
            self.vision.hidden, n_labels=len(LABELS),
            n_heads=model_cfg.get("cls_heads", 8),
            p_drop=model_cfg.get("cls_dropout", 0.1))

    def load_backbone(self, ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        loaded = 0
        for key in ("vision_lora", "vision_patch_embed"):
            if key in state:
                self.vision.load_state_dict(state[key], strict=False)
                loaded += len(state[key])
        return loaded

    def set_backbone_trainable(self, trainable):
        for p in self.vision.parameters():
            p.requires_grad_(False)
        if trainable:
            for name, p in self.vision.named_parameters():
                if "lora" in name or ("patch_embed" in name and self.vision.patch_embed_full):
                    p.requires_grad_(True)

    def forward(self, video, return_attn=False):
        trainable = any(p.requires_grad for p in self.vision.parameters())
        with torch.enable_grad() if trainable else torch.no_grad():
            tokens, _ = self.vision(video, None)
        return self.head(tokens, return_attn=return_attn)
