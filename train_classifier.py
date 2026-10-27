import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from evijepa.classifier import CTLabelClassifier, LABELS
from evijepa.data import CTRateLabelDataset, collate_labels


def positive_weight(dataset, cap):
    targets = np.stack([t for _, t in dataset.samples])
    positive = targets.sum(0)
    negative = len(targets) - positive
    weight = np.where(positive > 0, negative / np.maximum(positive, 1), 1.0)
    return torch.from_numpy(np.minimum(weight, cap).astype(np.float32))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probs, targets = [], []
    for video, target in tqdm(loader, desc="valid", leave=False):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(video.to(device))
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        targets.append(target.numpy())
    probs, targets = np.concatenate(probs), np.concatenate(targets)
    scores = [roc_auc_score(targets[:, j], probs[:, j])
              for j in range(targets.shape[1]) if len(set(targets[:, j])) > 1]
    model.train()
    return float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    data_cfg, model_cfg, train_cfg = cfg["data"], cfg["model"], cfg["train"]

    accelerator = Accelerator(
        mixed_precision="bf16" if train_cfg.get("bf16", True) else "no",
        gradient_accumulation_steps=data_cfg.get("gradient_accumulation_steps", 1),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])

    train_set = CTRateLabelDataset(data_cfg["train_dir"], data_cfg["labels"], LABELS,
                                   augment=data_cfg.get("augment"))
    valid_set = CTRateLabelDataset(data_cfg["valid_dir"], data_cfg["valid_labels"], LABELS)
    train_loader = DataLoader(train_set, batch_size=data_cfg["batch_size"], shuffle=True,
                              collate_fn=collate_labels, num_workers=data_cfg["num_workers"],
                              pin_memory=True, drop_last=True,
                              prefetch_factor=data_cfg.get("prefetch_factor", 2))
    valid_loader = DataLoader(valid_set, batch_size=data_cfg["batch_size"], shuffle=False,
                              collate_fn=collate_labels, num_workers=4)

    model = CTLabelClassifier(cfg)
    if model_cfg.get("backbone_ckpt"):
        model.load_backbone(model_cfg["backbone_ckpt"])
    model.set_backbone_trainable(not model_cfg.get("freeze_vision", False))

    pos_weight = positive_weight(train_set, model_cfg.get("pos_weight_cap", 10.0))
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    vision_params = [p for p in model.vision.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": head_params, "lr": train_cfg["lr_head"]},
         {"params": vision_params, "lr": train_cfg["lr_vision"]}],
        weight_decay=train_cfg["weight_decay"])
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    model, optimizer, train_loader, valid_loader = accelerator.prepare(
        model, optimizer, train_loader, valid_loader)
    pos_weight = pos_weight.to(accelerator.device)
    ckpt_dir = Path(train_cfg["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    step, warmup = 0, train_cfg["warmup_steps"]
    for epoch in range(train_cfg["epochs"]):
        model.train()
        running, count = 0.0, 0
        bar = tqdm(train_loader, disable=not accelerator.is_main_process, desc=f"epoch {epoch}")
        for video, target in bar:
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    logits = model(video)
                    loss = F.binary_cross_entropy_with_logits(
                        logits.float(), target.float(), pos_weight=pos_weight)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
                step += 1
                if step <= warmup:
                    for group, base in zip(optimizer.param_groups, base_lrs):
                        group["lr"] = base * step / warmup
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running += float(loss.detach())
            count += 1
            bar.set_postfix(loss=f"{running / max(count, 1):.4f}")

        if accelerator.is_main_process:
            module = accelerator.unwrap_model(model)
            torch.cuda.empty_cache()
            auroc = evaluate(module, valid_loader, accelerator.device)
            vision_state = module.vision.state_dict()
            torch.save({
                "head": module.head.state_dict(),
                "vision_lora": {k: v for k, v in vision_state.items() if "lora" in k},
                "vision_patch_embed": {k: v for k, v in vision_state.items()
                                       if "patch_embed" in k and "lora" not in k},
                "epoch": epoch,
                "cfg": cfg,
            }, ckpt_dir / f"epoch_{epoch}.pt")
            print(f"epoch {epoch} loss {running / max(count, 1):.4f} macro_auroc {auroc:.4f}")


if __name__ == "__main__":
    main()
