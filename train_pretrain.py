import argparse
from datetime import timedelta
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from evijepa.data import CTRateOrganDataset, collate_organ
from evijepa.model import EviJEPA


def build_optimizer(model, lr, text_lr_mult, vision_lr_mult, weight_decay):
    fast, text_slow, vision_slow = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("vision.model."):
            vision_slow.append(param)
        elif name.startswith("text_encoder.model."):
            text_slow.append(param)
        else:
            fast.append(param)
    return torch.optim.AdamW(
        [{"params": fast, "lr": lr},
         {"params": text_slow, "lr": lr * text_lr_mult},
         {"params": vision_slow, "lr": lr * vision_lr_mult}],
        weight_decay=weight_decay)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    data_cfg, model_cfg, train_cfg = cfg["data"], cfg["model"], cfg["train"]

    accelerator = Accelerator(
        mixed_precision="bf16" if train_cfg.get("bf16", True) else "no",
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True),
                         InitProcessGroupKwargs(timeout=timedelta(minutes=120))])

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["gemma_name"])
    dataset = CTRateOrganDataset(data_cfg["train_dir"], data_cfg["desc_info"],
                                 data_cfg["conc_info"], tokenizer,
                                 organs=data_cfg["organs"],
                                 max_txt_len=data_cfg["max_txt_len"])
    loader = DataLoader(dataset, batch_size=data_cfg["batch_size"], shuffle=True,
                        collate_fn=collate_organ, num_workers=data_cfg["num_workers"],
                        pin_memory=True, drop_last=True)

    model = EviJEPA(cfg)
    optimizer = build_optimizer(model, train_cfg["lr"], train_cfg["y_lr_mult"],
                                train_cfg["vjepa_lr_mult"], train_cfg["weight_decay"])
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    start_epoch, step = 0, 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.predictor.load_state_dict(state["predictor"])
        model.text_encoder.load_state_dict(state["text_encoder"])
        if "vision_lora" in state:
            model.vision.load_state_dict(state["vision_lora"], strict=False)
        if "vision_patch_embed" in state:
            model.vision.load_state_dict(state["vision_patch_embed"], strict=False)
        if "vision_moe" in state and model.use_moe:
            model.vision_moe.load_state_dict(state["vision_moe"])
            model.text_moe.load_state_dict(state["text_moe"])
        start_epoch, step = state["epoch"] + 1, state.get("step", 0)

    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    ckpt_dir = Path(train_cfg["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    warmup = train_cfg["warmup_steps"]

    for epoch in range(start_epoch, train_cfg["epochs"]):
        model.train()
        running, count = 0.0, 0
        bar = tqdm(loader, disable=not accelerator.is_main_process, desc=f"epoch {epoch}")
        for batch in bar:
            with accelerator.autocast():
                total, losses = model(batch, temperature=train_cfg["temperature"])
            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(total)
            accelerator.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
            step += 1
            if step <= warmup:
                for group, base in zip(optimizer.param_groups, base_lrs):
                    group["lr"] = base * step / warmup
            optimizer.step()
            running += float(total.detach())
            count += 1
            bar.set_postfix(loss=f"{running / max(count, 1):.4f}")

        if accelerator.is_main_process:
            module = accelerator.unwrap_model(model)
            state = {
                "predictor": module.predictor.state_dict(),
                "text_encoder": module.text_encoder.state_dict(),
                "epoch": epoch,
                "step": step,
                "cfg": cfg,
            }
            vision_state = module.vision.state_dict()
            state["vision_lora"] = {k: v for k, v in vision_state.items() if "lora" in k}
            state["vision_patch_embed"] = {k: v for k, v in vision_state.items()
                                           if "patch_embed" in k and "lora" not in k}
            if module.use_moe:
                state["vision_moe"] = module.vision_moe.state_dict()
                state["text_moe"] = module.text_moe.state_dict()
            torch.save(state, ckpt_dir / f"epoch_{epoch}.pt")
            print(f"epoch {epoch} avg_loss {running / max(count, 1):.4f}")


if __name__ == "__main__":
    main()
