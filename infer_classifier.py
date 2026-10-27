import argparse
import glob
import json
import os
from pathlib import Path

import torch
import yaml

from evijepa.classifier import CTLabelClassifier, LABELS
from evijepa.preprocess import load_ct, to_video

TTA_VIEWS = [[], [2], [3], [2, 3]]


def load_calibration(path, device):
    if path and Path(path).exists():
        table = json.load(open(path))
        return torch.tensor([float(table[label]) for label in LABELS],
                            dtype=torch.float32, device=device)
    return torch.zeros(len(LABELS), dtype=torch.float32, device=device)


@torch.no_grad()
def predict(model, video, bias, device, use_tta):
    video = video[None].to(device)
    views = TTA_VIEWS if use_tta else [[]]
    probs = 0.0
    for axes in views:
        view = torch.flip(video, axes) if axes else video
        with torch.autocast("cuda", dtype=torch.bfloat16):
            probs = probs + torch.sigmoid(model(view).float())
    probs = (probs / len(views))[0].clamp(1e-6, 1 - 1e-6)
    logit = torch.log(probs / (1 - probs))
    return torch.sigmoid(logit + bias).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--input", default=os.environ.get("INPUT_DIR", "/input"))
    parser.add_argument("--output", default=os.environ.get("OUTPUT_DIR", "/output"))
    parser.add_argument("--calibration", default="")
    parser.add_argument("--no-tta", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = yaml.safe_load(open(args.config))
    model = CTLabelClassifier(cfg).to(device).eval()
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.head.load_state_dict(state["head"])
    for key in ("vision_lora", "vision_patch_embed"):
        if key in state:
            model.vision.load_state_dict(state[key], strict=False)
    model.requires_grad_(False)

    bias = load_calibration(args.calibration, device)
    files = sorted(glob.glob(os.path.join(args.input, "*.nii.gz")))
    predictions = []
    for path in files:
        name = Path(path).name
        for suffix in (".nii.gz", ".nii"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        volume, spacing = load_ct(path)
        video = to_video(volume, spacing, device)
        probs = predict(model, video, bias, device, not args.no_tta)
        predictions.append({
            "input_image_name": name,
            "probabilities": {label: float(probs[i]) for i, label in enumerate(LABELS)},
        })

    payload = {
        "name": "Generated probabilities",
        "type": "Abnormality Classification",
        "version": {"major": 1, "minor": 0},
        "predictions": predictions,
    }
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions.json").write_text(json.dumps([{"outputs": [{"value": payload}]}]))
    print(f"wrote {out_dir / 'predictions.json'} ({len(predictions)} cases)")


if __name__ == "__main__":
    main()
