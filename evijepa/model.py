import torch
import torch.nn as nn
import torch.distributed as dist

from .vision_encoder import VJEPAOrganEncoder
from .predictor import OrganPredictor
from .text_encoder import TextEncoder
from .moe import ExpertChoiceMoE, route_diversity_loss, evidence_diversity_loss
from .losses import organ_infonce, evidence_infonce, text_hashes, dist_on


def gather_organ_tokens(tokens, flags, cap):
    sequences, indices = [], []
    for b in range(tokens.shape[0]):
        idx = flags[b].nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        if idx.numel() > cap:
            idx = idx[torch.randperm(idx.numel(), device=idx.device)[:cap]]
        sequences.append(tokens[b][idx])
        indices.append(b)
    if not sequences:
        return None
    longest = max(s.shape[0] for s in sequences)
    padded = tokens.new_zeros(len(sequences), longest, tokens.shape[-1])
    mask = tokens.new_zeros(len(sequences), longest)
    for i, s in enumerate(sequences):
        padded[i, :s.shape[0]] = s
        mask[i, :s.shape[0]] = 1.0
    return padded, mask, torch.tensor(indices, device=tokens.device)


class EviJEPA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        data_cfg, model_cfg = cfg["data"], cfg["model"]
        self.organs = data_cfg["organs"]
        self.n_organs = len(self.organs)
        self.batch_size = data_cfg["batch_size"]
        self.max_organ_tokens = model_cfg["max_organ_tokens"]

        self.vision = VJEPAOrganEncoder(
            model_cfg["vjepa21_ckpt"],
            arch=model_cfg["vision_arch"].split("_")[1],
            n_organs=self.n_organs,
            lora=model_cfg.get("vision_lora", True),
            lora_r=model_cfg.get("lora_r", 16),
            lora_alpha=model_cfg.get("lora_alpha", 32),
            lora_dropout=model_cfg.get("lora_dropout", 0.05),
            grad_ckpt=model_cfg.get("gradient_checkpointing", False),
            z_unshuffle=model_cfg.get("z_unshuffle", 2),
            patch_embed_full=model_cfg.get("patch_embed_full", True))

        self.predictor = OrganPredictor(
            model_cfg["llama_name"],
            n_keep_layers=model_cfg.get("n_keep_layers", 8),
            vis_dim=self.vision.hidden,
            shared_dim=model_cfg["shared_dim"],
            grad_ckpt=model_cfg.get("gradient_checkpointing", False),
            n_organs=self.n_organs)

        self.text_encoder = TextEncoder(
            model_cfg["gemma_name"],
            shared_dim=model_cfg["shared_dim"],
            grad_ckpt=model_cfg.get("gradient_checkpointing", False))

        moe_cfg = model_cfg.get("moe", {})
        self.use_moe = moe_cfg.get("use_moe", True)
        if self.use_moe:
            n_experts = moe_cfg.get("n_experts", 8)
            evidence_dim = moe_cfg.get("evidence_dim", 512)
            tau = moe_cfg.get("tau", 0.5)
            self.vision_moe = ExpertChoiceMoE(self.predictor.hidden, n_experts, evidence_dim, tau=tau)
            self.text_moe = ExpertChoiceMoE(self.text_encoder.hidden, n_experts, evidence_dim, tau=tau)
            self.lambda_ev = moe_cfg.get("lambda_ev", 1.0)
            self.lambda_rd = moe_cfg.get("lambda_rd", 0.02)
            self.lambda_ed = moe_cfg.get("lambda_ed", 0.02)
            self.sinkhorn_iters = moe_cfg.get("sinkhorn_iters", 5)
            self.sinkhorn_eps = moe_cfg.get("sinkhorn_eps", 0.05)

    def forward(self, batch, temperature=0.07):
        text_ids, text_mask = batch["text_ids"], batch["text_mask"]
        normal_flag, organ_texts, abnormal = batch["normal_flag"], batch["organ_texts"], batch["abnormal"]

        tokens, flags = self.vision(batch["video"], batch["mask"])
        device = tokens.device
        total, losses = 0.0, {}

        for o, name in enumerate(self.organs):
            gathered = gather_organ_tokens(tokens, flags[:, o], self.max_organ_tokens)
            has_anchor = gathered is not None and bool(abnormal[gathered[2], o].any())
            if dist_on():
                flag = torch.tensor([1.0 if has_anchor else 0.0], device=device)
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            if not has_anchor:
                continue

            vis, vis_mask, idxs = gathered
            pred, vis_hidden = self.predictor(
                vis.to(self.predictor.vis_w.dtype), vis_mask, return_tokens=True, organ_idx=o)
            tgt, text_hidden = self.text_encoder(
                text_ids[idxs, o], text_mask[idxs, o], return_tokens=True)

            normal = normal_flag[idxs, o].to(device).bool()
            texts = [organ_texts[b][o] for b in idxs.tolist()]
            hashes = text_hashes(texts, device)

            global_loss = organ_infonce(pred, tgt, normal, hashes, self.batch_size, temperature)
            if global_loss is not None:
                total = total + global_loss
                losses[name] = global_loss.detach()

            if self.use_moe:
                vision_evidence, vision_routing = self.vision_moe(vis_hidden, vis_mask)
                text_evidence, text_routing = self.text_moe(text_hidden, text_mask[idxs, o])
                local_loss = evidence_infonce(
                    vision_evidence, text_evidence, normal, hashes,
                    temperature, self.sinkhorn_iters, self.sinkhorn_eps)
                diversity = (route_diversity_loss(vision_routing, vis_mask)
                             + route_diversity_loss(text_routing, text_mask[idxs, o]))
                diversity = diversity + (evidence_diversity_loss(vision_evidence)
                                         + evidence_diversity_loss(text_evidence))
                if local_loss is not None:
                    total = total + self.lambda_ev * local_loss
                    losses[name + "_evidence"] = local_loss.detach()
                total = total + self.lambda_rd * diversity
                losses[name + "_diversity"] = diversity.detach()

        if not losses:
            zero = sum((p.sum() * 0.0) for p in self.parameters() if p.requires_grad)
            return zero, {}
        return total, losses

    @torch.no_grad()
    def encode_volume(self, video, mask):
        tokens, flags = self.vision(video, mask)
        outputs = {}
        for o in range(self.n_organs):
            gathered = gather_organ_tokens(tokens, flags[:, o], self.max_organ_tokens)
            if gathered is None:
                continue
            vis, vis_mask, idxs = gathered
            pred = self.predictor(vis.to(self.predictor.vis_w.dtype), vis_mask, organ_idx=o)
            outputs[self.organs[o]] = (pred, idxs)
        return outputs
