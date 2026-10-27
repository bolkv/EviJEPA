import argparse
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from evijepa.preprocess import HU_MAX, HU_MIN, TARGET_SHAPE, TARGET_SPACING, crop_or_pad, resample

ORGAN_IDS = {"lung": 1, "heart": 2, "esophagus": 3, "aorta": 4}

TS_GROUPS = {
    "lung": ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
             "lung_middle_lobe_right", "lung_lower_lobe_right"],
    "heart": ["heart", "heart_myocardium", "heart_atrium_left", "heart_atrium_right",
              "heart_ventricle_left", "heart_ventricle_right"],
    "esophagus": ["esophagus"],
    "aorta": ["aorta"],
}

ORGAN_KEYWORDS = {
    "lung": ["lung", "pulmonary", "bronch", "pleura", "parenchym", "lobe", "emphysem",
             "nodule", "consolidation", "atelectas", "ground glass", "ground-glass",
             "fibrotic", "fibrosis", "interlobular", "mosaic", "trachea"],
    "heart": ["heart", "cardiac", "cardiomegaly", "pericard", "coronary", "ventric", "atri"],
    "esophagus": ["esophag", "hiatal", "hernia"],
    "aorta": ["aorta", "aortic", "arterial wall", "atheroscler"],
}


def split_sentences(text):
    if not isinstance(text, str):
        return []
    parts = [s.strip() for s in re.split(r"(?<=[.])\s+", text.replace("\n", " ")) if s.strip()]
    return [p if p.endswith(".") else p + "." for p in parts]


def assign_organs(sentences):
    assigned = {organ: [] for organ in ORGAN_IDS}
    for sentence in sentences:
        lowered = sentence.lower()
        for organ, keywords in ORGAN_KEYWORDS.items():
            if any(k in lowered for k in keywords):
                assigned[organ].append(sentence)
    return {organ: " ".join(items) for organ, items in assigned.items() if items}


def build_organ_text_json(reports_csv, desc_out, conc_out):
    table = pd.read_csv(reports_csv)
    desc_info, conc_info = {}, {}
    for _, row in table.iterrows():
        study = re.sub(r"_\d+$", "", str(row["VolumeName"]).replace(".nii.gz", ""))
        findings = assign_organs(split_sentences(row.get("Findings_EN", "")))
        impressions = assign_organs(split_sentences(row.get("Impressions_EN", "")))
        if findings:
            desc_info[study] = findings
        if impressions:
            conc_info[study] = impressions
    Path(desc_out).write_text(json.dumps(desc_info))
    Path(conc_out).write_text(json.dumps(conc_info))
    return len(desc_info), len(conc_info)


def segment_organs(ct_path, cache_dir, fast=True):
    from totalsegmentator.map_to_binary import class_map
    from totalsegmentator.python_api import totalsegmentator

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / (Path(ct_path).name.replace(".nii.gz", "") + "_ts.nii.gz")
    if not out_path.exists():
        totalsegmentator(str(ct_path), str(out_path), task="total", fast=fast, ml=True, quiet=True)
    labels = np.asarray(nib.load(str(out_path)).dataobj).astype(np.int16)

    name_to_id = {name: idx for idx, name in class_map["total"].items()}
    organ_mask = np.zeros(labels.shape, dtype=np.int8)
    for organ, members in TS_GROUPS.items():
        ids = [name_to_id[m] for m in members if m in name_to_id]
        if not ids:
            continue
        hit = np.isin(labels, ids)
        organ_mask[hit] = ORGAN_IDS[organ]
    return organ_mask


def process_volume(ct_path, organ_mask, device):
    image = nib.load(str(ct_path))
    volume = np.asarray(image.dataobj).astype(np.float32)
    spacing = tuple(abs(float(z)) for z in image.header.get_zooms()[:3])
    volume = np.clip((volume - HU_MIN) / (HU_MAX - HU_MIN), 0.0, 1.0)

    ct = resample(torch.from_numpy(volume), spacing, TARGET_SPACING, "trilinear", device)
    ct = crop_or_pad(ct, TARGET_SHAPE, 0.0).clamp_(0, 1)

    mask = resample(torch.from_numpy(organ_mask.astype(np.float32)), spacing,
                    TARGET_SPACING, "nearest", device)
    mask = crop_or_pad(mask, TARGET_SHAPE, 0.0)
    return ct.numpy().astype(np.float16), mask.numpy().astype(np.int8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--reports_csv", default="")
    parser.add_argument("--desc_info", default="desc_info.json")
    parser.add_argument("--conc_info", default="conc_info.json")
    parser.add_argument("--seg_cache", default="seg_cache")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshard", type=int, default=1)
    parser.add_argument("--skip_masks", action="store_true")
    args = parser.parse_args()

    if args.reports_csv:
        n_desc, n_conc = build_organ_text_json(args.reports_csv, args.desc_info, args.conc_info)
        print(f"organ texts: {n_desc} findings studies, {n_conc} impression studies")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.ct_dir).rglob("*.nii.gz"))[args.shard::args.nshard]
    print(f"{len(files)} volumes (shard {args.shard}/{args.nshard})")

    for i, ct_path in enumerate(files, 1):
        target = out_dir / (ct_path.name.replace(".nii.gz", "") + ".npz")
        if target.exists():
            continue
        organ_mask = (np.zeros(nib.load(str(ct_path)).shape, dtype=np.int8)
                      if args.skip_masks else segment_organs(ct_path, args.seg_cache))
        ct, mask = process_volume(ct_path, organ_mask, args.device)
        tmp = target.with_suffix(".npz.tmp")
        np.savez_compressed(tmp, ct=ct, mask=mask)
        tmp.replace(target)
        if i % 25 == 0:
            print(f"  {i}/{len(files)}")
    print("done")


if __name__ == "__main__":
    main()
