"""Packed human-demo trajectories for offline imitation.

Each episode is one compressed ``.npz``: JPEG RGB and 16-bit depth at the
training resolution, plus the action taken from that observation, the goal
id, and the agent pose. One file per episode keeps the filesystem usable.
"""

from __future__ import annotations

import io
from typing import Dict, List

import numpy as np

from spikingnav.config import DEFAULT_CONFIG

STORE_IMAGE_SIZE = DEFAULT_CONFIG.image_size


def _jpeg(rgb: np.ndarray) -> bytes:
    from PIL import Image

    image = Image.fromarray(np.ascontiguousarray(rgb), mode="RGB")
    if image.size != (STORE_IMAGE_SIZE, STORE_IMAGE_SIZE):
        image = image.resize((STORE_IMAGE_SIZE, STORE_IMAGE_SIZE), Image.BILINEAR)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _depth_u16(depth: np.ndarray) -> np.ndarray:
    array = np.asarray(depth)
    if array.ndim == 3:
        array = array[..., 0]
    array = np.clip(array.astype(np.float32), 0.0, 1.0)
    if array.shape != (STORE_IMAGE_SIZE, STORE_IMAGE_SIZE):
        from PIL import Image

        image = Image.fromarray(array, mode="F")
        image = image.resize((STORE_IMAGE_SIZE, STORE_IMAGE_SIZE), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32)
    return np.round(array * 65535.0).astype(np.uint16)


def save_episode(path: str, record: Dict) -> None:
    rgbs = np.asarray([np.frombuffer(_jpeg(frame), dtype=np.uint8) for frame in record["rgb"]], dtype=object)
    depth = np.stack([_depth_u16(frame) for frame in record["depth"]], 0)
    np.savez_compressed(
        path,
        rgb_jpg=rgbs,
        depth_u16=depth,
        actions=np.asarray(record["actions"], dtype=np.int16),
        goal=np.int64(record["goal"]),
        episode_id=np.asarray(record["episode_id"]),
        scene_id=np.asarray(record["scene_id"]),
        object_category=np.asarray(record["object_category"]),
        positions=np.asarray(record["positions"], dtype=np.float32),
        rotations=np.asarray(record["rotations"], dtype=np.float32),
        geodesic_distance=np.float32(record.get("geodesic_distance", -1.0)),
    )


def load_episode(path: str) -> Dict:
    from PIL import Image

    with np.load(path, allow_pickle=True) as data:
        frames: List[np.ndarray] = []
        for raw in data["rgb_jpg"]:
            image = Image.open(io.BytesIO(raw.tobytes())).convert("RGB")
            frames.append(np.asarray(image, dtype=np.uint8))
        depth = data["depth_u16"].astype(np.float32) / 65535.0
        return {
            "rgb": np.stack(frames, 0),
            "depth": depth,
            "actions": data["actions"].astype(np.int64),
            "goal": int(data["goal"]),
            "episode_id": str(data["episode_id"]),
            "scene_id": str(data["scene_id"]),
            "object_category": str(data["object_category"]),
            "positions": data["positions"],
            "rotations": data["rotations"],
            "geodesic_distance": float(data["geodesic_distance"]),
        }
