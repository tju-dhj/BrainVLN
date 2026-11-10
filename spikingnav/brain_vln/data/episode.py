"""Load one collected episode without taking privileged navigation fields."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from spikingnav.brain_vln.data.contract import (
    ACTION_NAMES,
    ACTION_PAD,
    ALIGNMENT,
    FRAME_PATTERN,
    TRUNCATE_EVERY,
    DataContractError,
)
from spikingnav.brain_vln.data.transforms import SyncRGBDTransform


@dataclass
class EpisodeHeader:
    episode_dir: Path
    episode_id: str
    scene: str
    scene_id: str
    source_split: str
    steps: int
    depth_scale: float
    rgb_size: Tuple[int, int]
    depth_size: Tuple[int, int]
    instruction_text: str
    tokens: torch.Tensor
    token_mask: torch.Tensor
    actions: torch.Tensor


@dataclass
class EpisodeRecord:
    """One episode fragment. Coordinates and the goal stay out of this object."""

    episode_id: str
    scene: str
    scene_id: str
    source_split: str
    episode_steps: int
    window_start: int
    burn_in_steps: int
    instruction_text: str
    rgb: torch.Tensor
    depth: torch.Tensor
    depth_valid: torch.Tensor
    actions: torch.Tensor
    valid_mask: torch.Tensor
    loss_mask: torch.Tensor
    burn_in_mask: torch.Tensor
    episode_mask: torch.Tensor
    reset_mask: torch.Tensor
    detach_mask: torch.Tensor
    tokens: torch.Tensor
    token_mask: torch.Tensor
    subinstruction_spans: None = None


def frame_path(episode_dir: Path, modality: str, step: int) -> Path:
    """Address a frame by step index. Listing order is not time."""
    return episode_dir / modality / FRAME_PATTERN.format(step=step)


def _load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise DataContractError(f"could not read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DataContractError(f"{path} must contain a JSON object")
    return payload


def _assert_frame_files(episode_dir: Path, modality: str, steps: int) -> None:
    directory = episode_dir / modality
    expected = {FRAME_PATTERN.format(step=step) for step in range(steps)}
    if not directory.is_dir():
        raise DataContractError(f"missing {modality} directory in {episode_dir.name}")
    found = {path.name for path in directory.iterdir() if path.is_file()}
    if found != expected:
        missing = sorted(expected - found)[:8]
        extra = sorted(found - expected)[:8]
        raise DataContractError(
            f"{episode_dir.name} {modality} frame set mismatch; missing={missing} extra={extra}"
        )


def _read_rgb(path: Path, expected_size: Tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "RGB":
            raise DataContractError(f"{path.name} is not an RGB PNG")
        if list(image.size) != list(expected_size):
            raise DataContractError(f"{path.name} size {list(image.size)} != {list(expected_size)}")
        array = np.array(image)
    if array.dtype != np.uint8 or array.shape != (expected_size[1], expected_size[0], 3):
        raise DataContractError(f"{path.name} is not uint8 RGB of the declared size")
    return array


def _read_depth(path: Path, expected_size: Tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode not in ("I;16", "I;16B", "I"):
            raise DataContractError(f"{path.name} is not a 16-bit depth PNG")
        if list(image.size) != list(expected_size):
            raise DataContractError(f"{path.name} size {list(image.size)} != {list(expected_size)}")
        array = np.array(image)
    if array.ndim != 2 or array.shape != (expected_size[1], expected_size[0]):
        raise DataContractError(f"{path.name} depth shape does not match the declaration")
    raw = np.asarray(array, dtype=np.int64)
    if raw.min() < 0 or raw.max() > 65535:
        raise DataContractError(f"{path.name} depth values fall outside uint16")
    return raw


def _tokens_from_episode(episode_dir: Path, instruction: dict, episode_id: str, scene_id: str) -> torch.Tensor:
    if instruction.get("episode_id") != episode_id or instruction.get("scene_id") != scene_id:
        raise DataContractError(f"{episode_id} instruction identity does not match metadata")
    text = instruction.get("instruction_text")
    if not isinstance(text, str) or not text.strip():
        raise DataContractError(f"{episode_id} instruction_text is empty")
    sub_keys = [key for key in instruction if "sub" in key.lower() or "fsa" in key.lower()]
    if sub_keys:
        raise DataContractError(
            f"{episode_id} has {sub_keys}; P01 keeps subinstruction_spans empty and does not segment"
        )
    listed = instruction.get("instruction_tokens")
    if not isinstance(listed, list) or not listed or any(type(item) is not int for item in listed):
        raise DataContractError(f"{episode_id} instruction_tokens must be a non-empty int list")
    try:
        stored = np.load(episode_dir / "instruction" / "tokens.npy", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise DataContractError(f"{episode_id} tokens.npy is unreadable: {exc}") from exc
    if stored.ndim != 1 or not np.array_equal(stored, np.asarray(listed)):
        raise DataContractError(f"{episode_id} tokens.npy does not match instruction_tokens")
    zeros = np.flatnonzero(stored == 0)
    if len(zeros) and np.any(stored[zeros[0] :] != 0):
        raise DataContractError(f"{episode_id} padding token 0 occurs before a later token")
    if not np.any(stored != 0):
        raise DataContractError(f"{episode_id} instruction tokens are all padding")
    return torch.tensor(stored, dtype=torch.long)


def _assert_trajectory_alignment(episode_dir: Path, steps: int) -> None:
    payload = _load_json(episode_dir / "trajectories" / "trajectory.json")
    positions = payload.get("positions")
    if not isinstance(positions, list) or len(positions) != steps:
        raise DataContractError(f"{episode_dir.name} positions must have one pre-action pose per action")
    for key in ("locations", "reference_path"):
        if key not in payload:
            raise DataContractError(f"{episode_dir.name} trajectory is missing {key}")


def read_header(episode_dir: Path) -> EpisodeHeader:
    """Validate labels and frame names. Pixel arrays are loaded later."""
    episode_dir = Path(episode_dir)
    required = [
        episode_dir / "READY",
        episode_dir / "metadata.json",
        episode_dir / "action" / "actions.json",
        episode_dir / "instruction" / "instruction.json",
        episode_dir / "instruction" / "tokens.npy",
        episode_dir / "trajectories" / "trajectory.json",
        episode_dir / "rgb",
        episode_dir / "depth",
    ]
    missing = [str(path.relative_to(episode_dir)) for path in required if not path.exists()]
    if missing:
        raise DataContractError(f"{episode_dir.name} missing {missing}")

    metadata = _load_json(episode_dir / "metadata.json")
    action_record = _load_json(episode_dir / "action" / "actions.json")
    instruction = _load_json(episode_dir / "instruction" / "instruction.json")

    episode_id = str(metadata.get("episode_id", ""))
    scene = str(metadata.get("scene", ""))
    scene_id = str(metadata.get("scene_id", ""))
    if episode_id != episode_dir.name or scene != episode_dir.parent.name:
        raise DataContractError(f"{episode_dir} identity does not match its directory")
    if metadata.get("split") != "train":
        raise DataContractError(f"{episode_id} source split must be train")
    if metadata.get("alignment") != ALIGNMENT or metadata.get("frame_pattern") != FRAME_PATTERN:
        raise DataContractError(f"{episode_id} alignment or frame pattern is not the collected contract")

    try:
        steps = int(metadata["steps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DataContractError(f"{episode_id} has no integer step count") from exc
    if steps < 1:
        raise DataContractError(f"{episode_id} has no steps")

    names = action_record.get("action_names")
    if not isinstance(names, list) or names != list(ACTION_NAMES):
        raise DataContractError(f"{episode_id} action_names {names} != {list(ACTION_NAMES)}")
    actions_list = action_record.get("actions")
    if not isinstance(actions_list, list) or len(actions_list) != steps:
        raise DataContractError(f"{episode_id} actions length does not equal steps")
    if any(type(item) is not int for item in actions_list):
        raise DataContractError(f"{episode_id} actions must be ints")
    actions = torch.tensor(actions_list, dtype=torch.long)
    if torch.any((actions < 0) | (actions > 3)):
        raise DataContractError(f"{episode_id} action id falls outside 0..3")
    stops = torch.nonzero(actions == 0, as_tuple=False).flatten().tolist()
    if stops != [steps - 1]:
        raise DataContractError(f"{episode_id} must end with a single STOP, found indices {stops}")

    scale = metadata.get("depth_scale")
    if not isinstance(scale, (int, float)) or isinstance(scale, bool) or scale <= 0:
        raise DataContractError(f"{episode_id} depth_scale must be positive")
    if metadata.get("depth_unit") != "meter":
        raise DataContractError(f"{episode_id} depth_unit must be meter")
    rgb_size = _pair(metadata.get("rgb_size"), "rgb_size", episode_id)
    depth_size = _pair(metadata.get("depth_size"), "depth_size", episode_id)

    _assert_frame_files(episode_dir, "rgb", steps)
    _assert_frame_files(episode_dir, "depth", steps)
    _assert_trajectory_alignment(episode_dir, steps)
    tokens = _tokens_from_episode(episode_dir, instruction, episode_id, scene_id)
    text = str(instruction["instruction_text"])
    return EpisodeHeader(
        episode_dir=episode_dir,
        episode_id=episode_id,
        scene=scene,
        scene_id=scene_id,
        source_split="train",
        steps=steps,
        depth_scale=float(scale),
        rgb_size=rgb_size,
        depth_size=depth_size,
        instruction_text=text,
        tokens=tokens,
        token_mask=tokens != 0,
        actions=actions,
    )


def _pair(value, label: str, episode_id: str) -> Tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(item) is not int or item < 1 for item in value)
    ):
        raise DataContractError(f"{episode_id} {label} must be [width, height]")
    return int(value[0]), int(value[1])


def _sequence_masks(
    length: int,
    valid_length: int,
    burn_in_steps: int,
    truncate_every: int,
) -> Dict[str, torch.Tensor]:
    if truncate_every < 1:
        raise DataContractError("truncate_every must be positive")
    if not 0 <= burn_in_steps <= valid_length <= length:
        raise DataContractError("burn-in and valid length do not fit the padded window")
    valid = torch.zeros(length, dtype=torch.bool)
    valid[:valid_length] = True
    loss = torch.zeros(length, dtype=torch.bool)
    loss[burn_in_steps:valid_length] = True
    burn = torch.zeros(length, dtype=torch.bool)
    burn[:burn_in_steps] = True
    reset = torch.zeros(length, dtype=torch.bool)
    reset[0] = True
    detach = torch.zeros(length, dtype=torch.bool)
    for step in range(truncate_every, valid_length, truncate_every):
        detach[step] = True
    return {
        "valid_mask": valid,
        "loss_mask": loss,
        "burn_in_mask": burn,
        "episode_mask": valid.clone(),
        "reset_mask": reset,
        "detach_mask": detach,
    }


def _pad_time(tensor: torch.Tensor, length: int, value) -> torch.Tensor:
    if tensor.shape[0] == length:
        return tensor
    if tensor.shape[0] > length:
        raise DataContractError("cannot pad a tensor down")
    padding = tensor.new_full((length - tensor.shape[0], *tensor.shape[1:]), value)
    return torch.cat((tensor, padding), dim=0)


def load_episode_record(
    episode_dir: Path,
    *,
    start: int = 0,
    valid_length: Optional[int] = None,
    burn_in_steps: int = 0,
    pad_to: Optional[int] = None,
    truncate_every: int = TRUNCATE_EVERY,
    transform: Optional[SyncRGBDTransform] = None,
    header: Optional[EpisodeHeader] = None,
) -> EpisodeRecord:
    """Load steps ``[start, start + valid_length)`` addressed as ``{step}_0.png``."""
    header = header if header is not None else read_header(episode_dir)
    if valid_length is None:
        valid_length = header.steps - start
    if start < 0 or valid_length < 1 or start + valid_length > header.steps:
        raise DataContractError(
            f"{header.episode_id} window [{start}, {start + valid_length}) is outside 0..{header.steps}"
        )
    if burn_in_steps < 0 or burn_in_steps >= valid_length:
        raise DataContractError(f"{header.episode_id} burn-in leaves no supervised action")
    length = valid_length if pad_to is None else pad_to
    if length < valid_length:
        raise DataContractError(f"{header.episode_id} pad_to is shorter than the window")

    spatial = transform or SyncRGBDTransform()
    rgb_frames = []
    depth_frames = []
    valid_frames = []
    for step in range(start, start + valid_length):
        rgb = _read_rgb(frame_path(header.episode_dir, "rgb", step), header.rgb_size)
        depth_raw = _read_depth(frame_path(header.episode_dir, "depth", step), header.depth_size)
        rgb, depth_raw = spatial(rgb, depth_raw)
        rgb_frames.append(torch.tensor(np.transpose(rgb, (2, 0, 1)), dtype=torch.float32) / 255.0)
        raw_tensor = torch.tensor(depth_raw, dtype=torch.float32)
        valid = raw_tensor != 0
        meters = torch.zeros_like(raw_tensor)
        meters[valid] = raw_tensor[valid] / float(header.depth_scale)
        depth_frames.append(meters.unsqueeze(0))
        valid_frames.append(valid.unsqueeze(0))

    rgb_tensor = _pad_time(torch.stack(rgb_frames, dim=0), length, 0)
    depth_tensor = _pad_time(torch.stack(depth_frames, dim=0), length, 0)
    depth_valid = _pad_time(torch.stack(valid_frames, dim=0), length, False)
    actions = _pad_time(header.actions[start : start + valid_length].clone(), length, ACTION_PAD)
    masks = _sequence_masks(length, valid_length, burn_in_steps, truncate_every)
    return EpisodeRecord(
        episode_id=header.episode_id,
        scene=header.scene,
        scene_id=header.scene_id,
        source_split=header.source_split,
        episode_steps=header.steps,
        window_start=start,
        burn_in_steps=burn_in_steps,
        instruction_text=header.instruction_text,
        rgb=rgb_tensor,
        depth=depth_tensor,
        depth_valid=depth_valid,
        actions=actions,
        tokens=header.tokens.clone(),
        token_mask=header.token_mask.clone(),
        subinstruction_spans=None,
        **masks,
    )
