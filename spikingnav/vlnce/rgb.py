"""ImageNet normalization for RGB observations."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from spikingnav.vlnce.constants import IMAGENET_MEAN, IMAGENET_STD, RGB_SIZE


def prepare_rgb(rgb_uint8: torch.Tensor) -> torch.Tensor:
    """``[T, N, H, W, 3]`` uint8 to ImageNet-normalized ``[T, N, 3, H, W]``."""
    flat = rgb_uint8.reshape(-1, *rgb_uint8.shape[-3:]).float().permute(0, 3, 1, 2) / 255.0
    if flat.shape[-2] != RGB_SIZE or flat.shape[-1] != RGB_SIZE:
        flat = F.interpolate(flat, size=(RGB_SIZE, RGB_SIZE), mode="bilinear", align_corners=False)
    mean = flat.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = flat.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    flat = (flat - mean) / std
    return flat.reshape(*rgb_uint8.shape[:2], *flat.shape[1:])
