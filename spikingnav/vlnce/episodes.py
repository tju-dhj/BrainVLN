"""Pair R2R-CE episodes with the precomputed low-level expert actions."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

from spikingnav.vlnce.constants import PAD_INDEX


def load_json_gz(path: Path):
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def token_length(tokens: Iterable[int], pad_index: int = PAD_INDEX) -> int:
    values = list(tokens)
    end = len(values)
    while end > 0 and int(values[end - 1]) == pad_index:
        end -= 1
    return max(end, 1)


def pair_demonstration(episode: dict, gt: dict) -> dict:
    """One supervised trajectory: instruction tokens plus expert actions."""
    actions = [int(action) for action in gt["actions"]]
    if not actions:
        raise ValueError(f"episode {episode.get('episode_id')} has no expert actions")
    instruction = episode["instruction"]
    goals = episode.get("goals") or []
    if not goals:
        raise ValueError(f"episode {episode.get('episode_id')} has no goal")
    info = episode.get("info") or {}
    return {
        "episode_id": str(episode["episode_id"]),
        "trajectory_id": str(episode.get("trajectory_id", "")),
        "scene_id": str(episode["scene_id"]),
        "instruction_text": str(instruction.get("instruction_text", "")),
        "tokens": [int(token) for token in instruction["instruction_tokens"]],
        "actions": actions,
        "start_position": [float(v) for v in episode["start_position"]],
        "start_rotation": [float(v) for v in episode["start_rotation"]],
        "goal": [float(v) for v in goals[0]["position"]],
        "geodesic_distance": float(info.get("geodesic_distance", -1.0)),
        "reference_path": [[float(v) for v in point] for point in episode.get("reference_path") or []],
        "locations": [[float(v) for v in point] for point in gt.get("locations") or []],
    }


def iter_split(dataset_root: str, split: str) -> Iterator[dict]:
    root = Path(dataset_root)
    episodes = load_json_gz(root / split / f"{split}.json.gz")["episodes"]
    ground_truth = load_json_gz(root / split / f"{split}_gt.json.gz")
    for episode in episodes:
        key = str(episode["episode_id"])
        if key not in ground_truth:
            raise KeyError(f"{split} episode {key} is missing from the ground-truth file")
        yield pair_demonstration(episode, ground_truth[key])


def load_split(dataset_root: str, split: str, limit: int = 0) -> List[dict]:
    demos = []
    for demo in iter_split(dataset_root, split):
        demos.append(demo)
        if limit > 0 and len(demos) >= limit:
            break
    return demos


def mesh_path(scenes_dir: str, scene_id: str) -> Path:
    relative = scene_id
    prefix = "data/scene_datasets/"
    if relative.startswith(prefix):
        relative = relative[len(prefix):]
    return Path(scenes_dir) / relative


def missing_meshes(demos: Iterable[dict], scenes_dir: str) -> Dict[str, str]:
    missing = {}
    for demo in demos:
        path = mesh_path(scenes_dir, demo["scene_id"])
        if not path.is_file():
            missing[demo["episode_id"]] = str(path)
    return missing


def load_embeddings(path: str):
    import torch

    values = load_json_gz(Path(path))
    embeddings = torch.tensor(values, dtype=torch.float32)
    if embeddings.dim() != 2:
        raise ValueError(f"Expected an embedding matrix, got shape {tuple(embeddings.shape)}")
    return embeddings
