"""RobustNav-style visual corruptions applied to RGB frames."""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import cv2
import numpy as np
from PIL import Image


CORRUPTION_ALIASES = {
    "defocus_blur": "Defocus Blur",
    "defocus blur": "Defocus Blur",
    "motion_blur": "Motion Blur",
    "motion blur": "Motion Blur",
    "lighting": "Low Lighting",
    "low lighting": "Low Lighting",
    "low_lighting": "Low Lighting",
    "speckle_noise": "Speckle Noise",
    "speckle noise": "Speckle Noise",
    "spatter": "Spatter",
    "camera_crack": "Camera Crack",
    "camera crack": "Camera Crack",
    "lower_fov": "Lower FOV",
    "lower fov": "Lower FOV",
}


def _as_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.float32 or image.dtype == np.float64:
        if image.max() <= 1.5:
            image = image * 255.0
        image = np.clip(image, 0, 255)
    return np.asarray(image, dtype=np.uint8)


def _disk(radius: float, alias_blur: float = 0.1) -> np.ndarray:
    if radius <= 8:
        lim = np.arange(-8, 9)
        ksize = (3, 3)
    else:
        lim = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    x, y = np.meshgrid(lim, lim)
    kernel = ((x ** 2 + y ** 2) <= radius ** 2).astype(np.float32)
    kernel /= kernel.sum()
    return cv2.GaussianBlur(kernel, ksize=ksize, sigmaX=alias_blur)


def speckle_noise(x: np.ndarray, severity: int = 1) -> np.ndarray:
    scale = [0.15, 0.2, 0.35, 0.45, 0.6][severity - 1]
    x = _as_uint8(x).astype(np.float32) / 255.0
    noisy = x + x * np.random.normal(scale=scale, size=x.shape)
    return np.clip(noisy, 0, 1) * 255


def defocus_blur(x: np.ndarray, severity: int = 1) -> np.ndarray:
    radius, alias = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]
    x = _as_uint8(x).astype(np.float32) / 255.0
    kernel = _disk(radius, alias)
    blurred = np.stack(
        [cv2.filter2D(x[:, :, c], -1, kernel) for c in range(3)], axis=-1
    )
    return np.clip(blurred, 0, 1) * 255


def motion_blur(x: np.ndarray, severity: int = 1) -> np.ndarray:
    length = [10, 15, 15, 15, 20][severity - 1]
    if length % 2 == 0:
        length += 1
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0 / length
    angle = np.random.uniform(-45, 45)
    rot = cv2.getRotationMatrix2D((length / 2, length / 2), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (length, length))
    kernel /= max(kernel.sum(), 1e-6)
    x = _as_uint8(x)
    return np.clip(cv2.filter2D(x, -1, kernel), 0, 255)


def spatter(x: np.ndarray, severity: int = 1) -> np.ndarray:
    loc, scale, sigma, thresh, mud_sigma, mud = [
        (0.65, 0.3, 4, 0.69, 0.6, 0),
        (0.65, 0.3, 3, 0.68, 0.6, 0),
        (0.65, 0.3, 2, 0.68, 0.5, 0),
        (0.65, 0.3, 1, 0.65, 1.5, 1),
        (0.67, 0.4, 1, 0.65, 1.5, 1),
    ][severity - 1]
    x = _as_uint8(x).astype(np.float32) / 255.0
    layer = np.random.normal(loc=loc, scale=scale, size=x.shape[:2]).astype(np.float32)
    layer = cv2.GaussianBlur(layer, (0, 0), sigmaX=sigma)
    if mud == 0:
        mask = (layer > thresh).astype(np.float32)
        color = np.array([175, 238, 238], dtype=np.float32) / 255.0
        return np.clip(x + mask[..., None] * 0.35 * color, 0, 1) * 255
    mask = (layer > thresh).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=mud_sigma)
    mask[mask < 0.8] = 0
    color = np.array([63, 42, 20], dtype=np.float32) / 255.0
    x = x * (1.0 - mask[..., None]) + color * mask[..., None]
    return np.clip(x, 0, 1) * 255


def low_lighting(x: np.ndarray, severity: int = 1) -> np.ndarray:
    shift = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    hsv = cv2.cvtColor(_as_uint8(x), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] / 255.0 - shift, 0, 1) * 255.0
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def camera_crack(x: np.ndarray, severity: int = 1) -> np.ndarray:
    image = _as_uint8(x).copy()
    h, w = image.shape[:2]
    n_lines = [2, 3, 4, 6, 8][severity - 1]
    thickness = [1, 1, 2, 2, 3][severity - 1]
    rng = np.random.RandomState(severity * 17)
    for _ in range(n_lines):
        p1 = (rng.randint(0, w), rng.randint(0, h))
        p2 = (rng.randint(0, w), rng.randint(0, h))
        color = (int(rng.randint(0, 40)),) * 3
        cv2.line(image, p1, p2, color, thickness)
    return image


def lower_fov(x: np.ndarray, severity: int = 1) -> np.ndarray:
    """Center crop then letterbox, approximating a reduced field of view."""
    image = _as_uint8(x)
    h, w = image.shape[:2]
    keep = [0.85, 0.75, 0.65, 0.55, 0.45][severity - 1]
    ch, cw = int(h * keep), int(w * keep)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    crop = image[y0 : y0 + ch, x0 : x0 + cw]
    resized = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return resized


_IMPLS = {
    "Defocus Blur": defocus_blur,
    "Motion Blur": motion_blur,
    "Low Lighting": low_lighting,
    "Speckle Noise": speckle_noise,
    "Spatter": spatter,
    "Camera Crack": camera_crack,
    "Lower FOV": lower_fov,
}


def canonicalize(name: str) -> str:
    key = name.replace("-", " ").strip()
    return CORRUPTION_ALIASES.get(key.lower(), key)


def apply_corruption(
    image: Union[np.ndarray, Image.Image],
    corruption: str,
    severity: int = 5,
) -> np.ndarray:
    if isinstance(image, Image.Image):
        image = np.array(image)
    name = canonicalize(corruption)
    if name not in _IMPLS:
        raise KeyError(f"Unknown corruption {corruption!r}. Known: {list(_IMPLS)}")
    return np.uint8(np.clip(_IMPLS[name](image, severity), 0, 255))


def apply_corruption_sequence(
    frame: np.ndarray,
    corruptions: Sequence[str],
    severities: Optional[Sequence[int]] = None,
) -> np.ndarray:
    if severities is None:
        severities = [5] * len(corruptions)
    image = frame
    for name, sev in zip(corruptions, severities):
        image = apply_corruption(image, name, sev)
    return np.array(image)
