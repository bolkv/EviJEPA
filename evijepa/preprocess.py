import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

TARGET_SPACING = (1.0, 1.0, 2.5)
TARGET_SHAPE = (384, 384, 128)
HU_MIN, HU_MAX = -1150.0, 350.0


def resample(volume, current_spacing, target_spacing, mode="trilinear", device="cpu"):
    nx = max(1, round(volume.shape[0] * current_spacing[0] / target_spacing[0]))
    ny = max(1, round(volume.shape[1] * current_spacing[1] / target_spacing[1]))
    nz = max(1, round(volume.shape[2] * current_spacing[2] / target_spacing[2]))
    kwargs = dict(mode=mode) if mode == "nearest" else dict(mode=mode, align_corners=False)
    out = F.interpolate(volume[None, None].to(device), size=(nx, ny, nz), **kwargs)
    return out[0, 0].cpu()


def crop_or_pad(volume, target, pad_value=0.0):
    out = volume
    for axis, size in enumerate(target):
        cur = out.shape[axis]
        if cur > size:
            start = (cur - size) // 2
            out = out.index_select(axis, torch.arange(start, start + size, device=out.device))
        elif cur < size:
            before = (size - cur) // 2
            after = size - cur - before
            pad = [0, 0, 0, 0, 0, 0]
            pad[(2 - axis) * 2] = before
            pad[(2 - axis) * 2 + 1] = after
            out = F.pad(out[None, None], pad, value=pad_value)[0, 0]
    return out


def load_ct(path):
    image = nib.load(str(path))
    volume = np.asarray(image.dataobj).astype(np.float32)
    spacing = tuple(abs(float(z)) for z in image.header.get_zooms()[:3])
    volume = np.clip((volume - HU_MIN) / (HU_MAX - HU_MIN), 0.0, 1.0)
    return volume, spacing


def to_video(volume, spacing, device="cpu"):
    tensor = resample(torch.from_numpy(volume), spacing, TARGET_SPACING, "trilinear", device)
    tensor = crop_or_pad(tensor, TARGET_SHAPE, 0.0).clamp_(0, 1)
    return tensor.permute(2, 0, 1).contiguous()


def preprocess(path, device="cpu"):
    volume, spacing = load_ct(path)
    return to_video(volume, spacing, device)
