#!/usr/bin/env python3
"""Walk Habitat-Web human demonstrations and store RGB, depth, and labels.

The human set is ``datasets/objectnav_hm3d_hd`` (76,394 train episodes).
Standard HM3D / Gibson / MP3D ObjectNav episodes have no human action trace.
Images are resized to 224 and packed one ``.npz`` per episode.
"""

from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict

import numpy as np

from spikingnav.habitat_objectnav.expert import demo_action_index
from spikingnav.habitat_objectnav.make_env import (
    DEFAULT_SCENES_DIR,
    build_habitat_config,
    load_dataset,
    scene_config_path,
)
from spikingnav.habitat_objectnav.offline_store import save_episode

HUMAN_ROOT = os.path.abspath("datasets/objectnav_hm3d_hd/objectnav_hm3d_hd")


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)[:180]


def _action_names(env) -> list:
    names = []
    index = 0
    while True:
        try:
            names.append(env.task.get_action_name(index))
        except (ValueError, IndexError):
            return names
        index += 1


def _pose(env):
    state = env.sim.get_agent_state()
    position = np.asarray(state.position, dtype=np.float32)
    rotation = state.rotation
    quat = np.asarray(
        [rotation.x, rotation.y, rotation.z, rotation.w], dtype=np.float32
    )
    return position, quat


def collect_scene(episodes, output_dir: str, gpu: int, split: str, max_steps: int) -> int:
    from habitat.core.env import Env
    from habitat.config import read_write

    scene = episodes[0].scene_id
    content_id = os.path.basename(scene).split(".")[0]
    config = build_habitat_config(
        split=split,
        scenes=[content_id],
        gpu_device_id=gpu,
        max_episode_steps=max_steps,
        episode_root=HUMAN_ROOT,
        scenes_dir=DEFAULT_SCENES_DIR,
        seed=0,
    )
    scene_config = scene_config_path(DEFAULT_SCENES_DIR)
    dataset = load_dataset(config, scene_config)
    by_id = {episode.episode_id: episode for episode in dataset.episodes}
    chosen = [by_id[episode.episode_id] for episode in episodes if episode.episode_id in by_id]
    if not chosen:
        return 0
    with read_write(config):
        config.habitat.simulator.scene = chosen[0].scene_id
        config.habitat.simulator.scene_dataset = scene_config
    env = Env(config=config.habitat, dataset=dataset)
    names = _action_names(env)
    scene_dir = os.path.join(output_dir, split, _safe(os.path.basename(os.path.dirname(scene))))
    os.makedirs(scene_dir, exist_ok=True)
    written = 0
    try:
        for episode in chosen:
            path = os.path.join(scene_dir, _safe(str(episode.episode_id)) + ".npz")
            if os.path.exists(path):
                continue
            env.current_episode = episode
            obs = env.reset()
            rgbs, depths, actions, positions, rotations = [], [], [], [], []
            goal = int(np.asarray(obs["objectgoal"]).reshape(-1)[0])
            elapsed = 0
            while elapsed < max_steps:
                action, valid = demo_action_index(
                    getattr(episode, "reference_replay", None), elapsed, names
                )
                if not valid:
                    break
                position, quat = _pose(env)
                rgb = np.asarray(obs["rgb"])
                if rgb.shape[-1] == 4:
                    rgb = rgb[..., :3]
                depth = np.asarray(obs["depth"])
                rgbs.append(rgb)
                depths.append(depth)
                actions.append(action)
                positions.append(position)
                rotations.append(quat)
                obs = env.step(int(action))
                elapsed += 1
                if env.episode_over:
                    break
            if not actions:
                continue
            info = getattr(episode, "info", None) or {}
            save_episode(
                path,
                {
                    "rgb": rgbs,
                    "depth": depths,
                    "actions": actions,
                    "goal": goal,
                    "episode_id": str(episode.episode_id),
                    "scene_id": str(episode.scene_id),
                    "object_category": str(getattr(episode, "object_category", "")),
                    "positions": positions,
                    "rotations": rotations,
                    "geodesic_distance": float(info.get("geodesic_distance", -1.0))
                    if isinstance(info, dict)
                    else -1.0,
                },
            )
            written += 1
            print(f"saved {path} steps {len(actions)}", flush=True)
    finally:
        env.close()
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=("train", "val"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", default="storage/il_offline/hm3d_hd")
    parser.add_argument("--max-episode-steps", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=0, help="Stop after this many new episodes. 0 means all.")
    args = parser.parse_args()
    config = build_habitat_config(
        split=args.split,
        gpu_device_id=args.gpu,
        max_episode_steps=args.max_episode_steps,
        episode_root=HUMAN_ROOT,
        scenes_dir=DEFAULT_SCENES_DIR,
    )
    dataset = load_dataset(config, scene_config_path(DEFAULT_SCENES_DIR))
    groups = defaultdict(list)
    for episode in dataset.episodes:
        groups[episode.scene_id].append(episode)
    done = 0
    for scene, episodes in groups.items():
        if args.limit and done >= args.limit:
            break
        batch = episodes
        if args.limit:
            batch = episodes[: max(0, args.limit - done)]
        done += collect_scene(batch, args.output_dir, args.gpu, args.split, args.max_episode_steps)
    print(f"wrote {done} episodes", flush=True)


if __name__ == "__main__":
    main()
