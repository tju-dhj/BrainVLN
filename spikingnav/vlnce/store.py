"""Per-episode RGB-D-instruction-action trees for R2R-CE expert rollouts."""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from spikingnav.vlnce.constants import (
    ACTION_NAMES,
    AGENT_HEIGHT,
    AGENT_RADIUS,
    DEPTH_SCALE,
    FORWARD_STEP,
    RGB_SIZE,
    SENSOR_HEIGHT,
    SUCCESS_DISTANCE,
    TURN_ANGLE,
)

READY_NAME = "READY"
CAMERA_INDEX = 0


def scene_name(scene_id: str) -> str:
    return Path(str(scene_id)).stem


def episode_dir(output: str, split: str, scene_id: str, episode_id: str) -> Path:
    return Path(output) / split / scene_name(scene_id) / str(episode_id)


def frame_name(step: int, camera: int = CAMERA_INDEX) -> str:
    return f"{int(step)}_{int(camera)}.png"


def _rgb_image(frame: np.ndarray) -> Image.Image:
    image = Image.fromarray(np.ascontiguousarray(np.asarray(frame)[..., :3], dtype=np.uint8))
    if image.size != (RGB_SIZE, RGB_SIZE):
        image = image.resize((RGB_SIZE, RGB_SIZE), Image.BILINEAR)
    return image


def _depth_meters(depth: np.ndarray) -> np.ndarray:
    array = np.asarray(depth, dtype=np.float32)
    if array.ndim == 3:
        array = array[..., 0]
    if array.shape != (RGB_SIZE, RGB_SIZE):
        resized = Image.fromarray(array, mode="F").resize((RGB_SIZE, RGB_SIZE), Image.BILINEAR)
        array = np.asarray(resized, dtype=np.float32)
    return np.nan_to_num(array, nan=0.0, posinf=65.535, neginf=0.0)


def _depth_png(depth: np.ndarray) -> Image.Image:
    scaled = np.clip(_depth_meters(depth) * DEPTH_SCALE, 0, 65535).astype(np.uint16)
    return Image.fromarray(scaled)


def _write_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def save_rollout(output: str, record: Dict, split: str = "train") -> Path:
    """Write one episode under ``output/split/scene/episode`` and return that directory."""
    final = episode_dir(output, split, record["scene_id"], record["episode_id"])
    if (final / READY_NAME).is_file():
        return final
    partial = final.parent / f"{record['episode_id']}.partial"
    if partial.exists():
        shutil.rmtree(partial)
    if final.exists():
        shutil.rmtree(final)
    partial.mkdir(parents=True)
    rgb_dir = partial / "rgb"
    depth_dir = partial / "depth"
    action_dir = partial / "action"
    instruction_dir = partial / "instruction"
    trajectory_dir = partial / "trajectories"
    for directory in (rgb_dir, depth_dir, action_dir, instruction_dir, trajectory_dir):
        directory.mkdir()

    frames = list(record["rgb"])
    depths = list(record["depth"])
    actions = [int(action) for action in record["actions"]]
    positions = np.asarray(record["positions"], dtype=np.float32)
    instruction_text = str(record.get("instruction_text", "")).strip()
    if not instruction_text:
        raise ValueError(f"episode {record.get('episode_id')} has no instruction text")
    if not (len(frames) == len(depths) == len(actions) == len(positions)):
        raise ValueError(
            f"episode {record.get('episode_id')} length mismatch "
            f"rgb={len(frames)} depth={len(depths)} actions={len(actions)} positions={len(positions)}"
        )
    for step, frame in enumerate(frames):
        name = frame_name(step)
        _rgb_image(frame).save(rgb_dir / name, format="PNG", compress_level=1)
        _depth_png(depths[step]).save(depth_dir / name, format="PNG", compress_level=1)

    tokens = [int(token) for token in record["tokens"]]
    _write_json(action_dir / "actions.json", {
        "action_names": list(ACTION_NAMES),
        "actions": actions,
    })
    _write_json(instruction_dir / "instruction.json", {
        "episode_id": str(record["episode_id"]),
        "scene_id": str(record["scene_id"]),
        "instruction_text": str(record.get("instruction_text", "")),
        "instruction_tokens": tokens,
    })
    np.save(instruction_dir / "tokens.npy", np.asarray(tokens, dtype=np.int32))
    _write_json(trajectory_dir / "trajectory.json", {
        "positions": positions.tolist(),
        "locations": np.asarray(record.get("locations", []), dtype=np.float32).tolist(),
        "reference_path": np.asarray(record.get("reference_path", []), dtype=np.float32).tolist(),
    })
    _write_json(partial / "metadata.json", {
        "episode_id": str(record["episode_id"]),
        "scene_id": str(record["scene_id"]),
        "scene": scene_name(record["scene_id"]),
        "split": split,
        "goal": np.asarray(record["goal"], dtype=np.float32).tolist(),
        "geodesic_distance": float(record.get("geodesic_distance", -1)),
        "start_position": np.asarray(record["start_position"], dtype=np.float32).tolist(),
        "start_rotation": np.asarray(record["start_rotation"], dtype=np.float32).tolist(),
        "steps": len(actions),
        "rgb_format": "PNG",
        "rgb_size": [RGB_SIZE, RGB_SIZE],
        "depth_format": "uint16_png",
        "depth_size": [RGB_SIZE, RGB_SIZE],
        "depth_scale": DEPTH_SCALE,
        "depth_unit": "meter",
        "frame_pattern": "{step}_0.png",
        "forward_step_m": FORWARD_STEP,
        "turn_angle_deg": TURN_ANGLE,
        "success_distance_m": SUCCESS_DISTANCE,
        "sensor_height_m": SENSOR_HEIGHT,
        "agent_height_m": AGENT_HEIGHT,
        "agent_radius_m": AGENT_RADIUS,
        "alignment": "rgb[t] and depth[t] are the observation before action[t]",
    })
    (partial / READY_NAME).write_text("ok\n")
    partial.rename(final)
    return final


def find_rollouts(root: str) -> List[str]:
    base = Path(root)
    if not base.exists():
        return []
    ready = sorted(path.parent for path in base.rglob(READY_NAME) if path.is_file())
    if ready:
        return [str(path) for path in ready]
    return [str(path) for path in sorted(base.glob("*.npz"))]


def _load_depth(path: Path, scale: float) -> np.ndarray:
    array = np.asarray(Image.open(path))
    if array.ndim == 3:
        array = array[..., 0]
    return array.astype(np.float32) / float(scale)


def load_rollout(path: str) -> Dict:
    root = Path(path)
    if root.is_file() and root.suffix == ".npz":
        return _load_npz(root)
    if root.name == READY_NAME:
        root = root.parent
    metadata = json.loads((root / "metadata.json").read_text())
    instruction = json.loads((root / "instruction" / "instruction.json").read_text())
    actions = json.loads((root / "action" / "actions.json").read_text())["actions"]
    trajectory = json.loads((root / "trajectories" / "trajectory.json").read_text())
    scale = float(metadata.get("depth_scale", DEPTH_SCALE))
    rgb = []
    depth = []
    for step in range(len(actions)):
        name = frame_name(step)
        rgb.append(np.asarray(Image.open(root / "rgb" / name).convert("RGB"), dtype=np.uint8))
        depth.append(_load_depth(root / "depth" / name, scale))
    return {
        "rgb": np.stack(rgb, 0),
        "depth": np.stack(depth, 0),
        "actions": np.asarray(actions, dtype=np.int64),
        "tokens": np.asarray(instruction["instruction_tokens"], dtype=np.int64),
        "instruction_text": str(instruction.get("instruction_text", "")),
        "positions": np.asarray(trajectory["positions"], dtype=np.float32),
        "episode_id": str(metadata["episode_id"]),
        "scene_id": str(metadata["scene_id"]),
        "goal": np.asarray(metadata["goal"], dtype=np.float32),
        "geodesic_distance": float(metadata["geodesic_distance"]),
        "start_position": np.asarray(metadata["start_position"], dtype=np.float32),
        "start_rotation": np.asarray(metadata["start_rotation"], dtype=np.float32),
        "reference_path": np.asarray(trajectory.get("reference_path", []), dtype=np.float32),
        "locations": np.asarray(trajectory.get("locations", []), dtype=np.float32),
    }


def _load_npz(path: Path) -> Dict:
    with np.load(path, allow_pickle=True) as data:
        frames: List[np.ndarray] = []
        for raw in data["rgb_jpg"]:
            blob = raw.tobytes() if isinstance(raw, np.ndarray) else bytes(raw)
            image = Image.open(io.BytesIO(blob)).convert("RGB")
            frames.append(np.asarray(image, dtype=np.uint8))
        reference = data["reference_path"] if "reference_path" in data.files else np.zeros((0, 3))
        locations = data["locations"] if "locations" in data.files else np.zeros((0, 3))
        return {
            "rgb": np.stack(frames, 0),
            "depth": data["depth"].astype(np.float32),
            "actions": data["actions"].astype(np.int64),
            "tokens": data["tokens"].astype(np.int64),
            "positions": data["positions"].astype(np.float32),
            "episode_id": str(data["episode_id"]),
            "scene_id": str(data["scene_id"]),
            "goal": data["goal"].astype(np.float32),
            "geodesic_distance": float(data["geodesic"]),
            "start_position": data["start_position"].astype(np.float32),
            "start_rotation": data["start_rotation"].astype(np.float32),
            "reference_path": reference.astype(np.float32),
            "locations": locations.astype(np.float32),
        }
