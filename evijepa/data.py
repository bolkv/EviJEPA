import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

ORGANS = ("lung", "heart", "esophagus", "aorta")


def normal_template(organ):
    return f"{organ} shows no significant abnormalities."


def build_organ_texts(study, desc_info, conc_info, organs):
    desc, conc = desc_info.get(study, {}), conc_info.get(study, {})
    texts = {}
    for organ in organs:
        description = desc.get(organ, "")
        if description and not description.endswith("."):
            description += "."
        conclusion = conc.get(organ, "")
        if conclusion and not conclusion.endswith("."):
            conclusion += "."
        if not conclusion:
            conclusion = normal_template(organ)
        text = conclusion + description
        for ch in '"\'()':
            text = text.replace(ch, "")
        texts[organ] = text
    return texts


def random_value(low, high):
    return low + (high - low) * torch.rand(()).item()


def spatial_augment(video, cfg):
    if torch.rand(()).item() < cfg.get("flip_p", 0.0):
        video = torch.flip(video, dims=[2])
    if torch.rand(()).item() < cfg.get("affine_p", 0.0):
        angle = math.radians(random_value(-cfg.get("rot_deg", 15), cfg.get("rot_deg", 15)))
        zoom = random_value(*cfg.get("zoom_range", [0.9, 1.1]))
        shift = cfg.get("translate", 0) / max(video.shape[-1], 1) * 2
        tx = random_value(-shift, shift)
        ty = random_value(-shift, shift)
        cos, sin = math.cos(angle) / zoom, math.sin(angle) / zoom
        theta = torch.tensor([[cos, -sin, tx], [sin, cos, ty]], dtype=torch.float32)[None]
        Z, H, W = video.shape
        grid = F.affine_grid(theta, (1, 1, H, W), align_corners=False).expand(Z, -1, -1, -1)
        video = F.grid_sample(video.unsqueeze(1), grid, mode="bilinear",
                              padding_mode="zeros", align_corners=False).squeeze(1)
    if torch.rand(()).item() < cfg.get("noise_p", 0.0):
        video = video + torch.randn_like(video) * cfg.get("noise_std", 0.02)
    return video.clamp(0, 1).contiguous()


class CTRateOrganDataset(Dataset):
    def __init__(self, data_dir, desc_info_path, conc_info_path, tokenizer,
                 organs=ORGANS, max_txt_len=384):
        self.data_dir = Path(data_dir)
        self.organs = list(organs)
        self.tokenizer = tokenizer
        self.max_txt_len = max_txt_len
        desc_info = json.load(open(desc_info_path))
        conc_info = json.load(open(conc_info_path))
        self.samples = []
        for path in sorted(self.data_dir.glob("*.npz")):
            study = re.sub(r"_\d+$", "", path.stem)
            if study in desc_info or study in conc_info:
                texts = build_organ_texts(study, desc_info, conc_info, self.organs)
                self.samples.append((str(path), study, texts))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, study, texts = self.samples[index]
        data = np.load(path)
        video = torch.from_numpy(np.transpose(data["ct"].astype(np.float32), (2, 0, 1)))
        mask = torch.from_numpy(np.transpose(data["mask"].astype(np.int8), (2, 0, 1)))

        organ_texts = [texts[o] for o in self.organs]
        encoded = self.tokenizer(organ_texts, padding="max_length", truncation=True,
                                 max_length=self.max_txt_len, return_tensors="pt")
        normal_flag = torch.tensor(
            [organ_texts[i].startswith(normal_template(o)) for i, o in enumerate(self.organs)])
        present = torch.tensor([(mask == i + 1).any().item() for i in range(len(self.organs))])

        return {
            "video": video,
            "mask": mask,
            "text_ids": encoded["input_ids"],
            "text_mask": encoded["attention_mask"],
            "organ_texts": organ_texts,
            "normal_flag": normal_flag,
            "present": present,
            "abnormal": present & (~normal_flag),
            "study": study,
        }


def collate_organ(batch):
    return {
        "video": torch.stack([b["video"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "text_ids": torch.stack([b["text_ids"] for b in batch]),
        "text_mask": torch.stack([b["text_mask"] for b in batch]),
        "normal_flag": torch.stack([b["normal_flag"] for b in batch]),
        "present": torch.stack([b["present"] for b in batch]),
        "abnormal": torch.stack([b["abnormal"] for b in batch]),
        "organ_texts": [b["organ_texts"] for b in batch],
        "study": [b["study"] for b in batch],
    }


class CTRateLabelDataset(Dataset):
    def __init__(self, data_dir, labels_csv, labels, augment=None):
        self.data_dir = Path(data_dir)
        self.labels = list(labels)
        self.augment = augment or None
        table = pd.read_csv(labels_csv)
        volumes = table.iloc[:, 0].str.replace(".nii.gz", "", regex=False)
        matrix = table[self.labels].to_numpy(np.float32)
        lookup = {v: matrix[i] for i, v in enumerate(volumes)}
        self.samples = [(str(p), lookup[p.stem]) for p in sorted(self.data_dir.glob("*.npz"))
                        if p.stem in lookup]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        video = torch.from_numpy(np.transpose(np.load(path)["ct"].astype(np.float32), (2, 0, 1)))
        if self.augment:
            video = spatial_augment(video, self.augment)
        return video.half(), torch.from_numpy(target)


def collate_labels(batch):
    videos = torch.stack([b[0] for b in batch])
    targets = torch.stack([b[1] for b in batch])
    return videos, targets
