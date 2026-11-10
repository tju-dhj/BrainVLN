"""Spatial transforms applied jointly to an RGB frame and its depth frame."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import Image

from spikingnav.brain_vln.data.contract import DataContractError


class SyncRGBDTransform:
    """Resize RGB and depth with the same output geometry, or leave both native.

    Depth uses nearest-neighbor sampling so an invalid raw value of 0 does not
    bleed into neighboring pixels. RGB uses bilinear sampling when a size is set.
    """

    def __init__(self, size: Optional[Tuple[int, int]] = None) -> None:
        if size is not None and (len(size) != 2 or size[0] < 1 or size[1] < 1):
            raise DataContractError("size must be a positive (height, width) pair")
        self.size = size

    def __call__(self, rgb: np.ndarray, depth_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise DataContractError("RGB transform expects uint8 HWC")
        if depth_raw.ndim != 2:
            raise DataContractError("depth transform expects a 2D raw array")
        if rgb.shape[:2] != depth_raw.shape[:2]:
            raise DataContractError(
                f"RGB shape {rgb.shape[:2]} and depth shape {depth_raw.shape[:2]} differ"
            )
        if self.size is None:
            return rgb, np.array(depth_raw, dtype=np.int64, copy=True)

        height, width = self.size
        rgb_image = Image.fromarray(rgb, mode="RGB").resize(
            (width, height),
            resample=Image.Resampling.BILINEAR,
        )
        raw = np.array(depth_raw)
        if raw.min() < 0 or raw.max() > 65535:
            raise DataContractError("depth raw values do not fit uint16")
        depth_image = Image.fromarray(raw.astype(np.uint16), mode="I;16").resize(
            (width, height),
            resample=Image.Resampling.NEAREST,
        )
        return np.array(rgb_image, dtype=np.uint8), np.array(depth_image, dtype=np.int64)
