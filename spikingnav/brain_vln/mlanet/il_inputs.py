"""Inputs for official VLN-CE CMA and Seq2Seq checkpoints.

Those weights were trained on a 224 RGB image and a 256 depth image.
Habitat now renders that same R2R-CE pair. The resize below is a no-op
when the incoming tensors already have those sizes.
"""

import numpy as np
import torch
import torch.nn.functional as F
from gym import spaces

VLNCE_IL_RGB = 224
VLNCE_IL_DEPTH = 256


def il_observation_space():
    """Shapes the official depth ResNet was built with, not the Habitat camera."""
    return spaces.Dict(
        {
            "rgb": spaces.Box(
                low=0,
                high=255,
                shape=(VLNCE_IL_RGB, VLNCE_IL_RGB, 3),
                dtype=np.uint8,
            ),
            "depth": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(VLNCE_IL_DEPTH, VLNCE_IL_DEPTH, 1),
                dtype=np.float32,
            ),
        }
    )


def _square(image, size: int, mode: str) -> torch.Tensor:
    if image.shape[1] == size and image.shape[2] == size:
        return image
    channels_first = image.permute(0, 3, 1, 2).float()
    if mode == "nearest":
        resized = F.interpolate(channels_first, size=(size, size), mode="nearest")
    else:
        resized = F.interpolate(
            channels_first, size=(size, size), mode="bilinear", align_corners=False
        )
    return resized.permute(0, 2, 3, 1)


def prepare_il_visuals(observations):
    """Return RGB at 224 and depth at 256, leaving instruction tensors untouched."""
    prepared = dict(observations)
    rgb = observations["rgb"]
    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError(f"expected RGB (B, H, W, 3), got {tuple(rgb.shape)}")
    depth = observations["depth"]
    if depth.ndim == 3:
        depth = depth.unsqueeze(-1)
    if depth.ndim != 4:
        raise ValueError(f"expected depth (B, H, W, C), got {tuple(depth.shape)}")
    prepared["rgb"] = _square(rgb, VLNCE_IL_RGB, "bilinear")
    prepared["depth"] = _square(depth[..., :1], VLNCE_IL_DEPTH, "nearest")
    return prepared
