"""Build a Habitat 0.3 ObjectNav environment on the local HM3D scenes."""

from __future__ import annotations

import json
import math
import os
from typing import List, Optional, Sequence

from spikingnav.config import DEFAULT_CONFIG

# HM3D ObjectNav v1 task categories (Habitat challenge / PONI-style episodes).
HM3D_CATEGORIES = ("chair", "bed", "plant", "toilet", "tv_monitor", "sofa")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_EPISODE_ROOT = os.path.join(REPO_ROOT, "datasets", "objectnav", "hm3d", "v1")
DEFAULT_SCENES_DIR = os.path.join(REPO_ROOT, "datasets", "scene_datasets")
SCENE_CONFIG_NAME = "hm3d_spikingnav.scene_dataset_config.json"


def scene_config_path(scenes_dir: str = DEFAULT_SCENES_DIR) -> str:
    return os.path.join(scenes_dir, "hm3d", SCENE_CONFIG_NAME)


def ensure_scene_dataset_config(scenes_dir: str = DEFAULT_SCENES_DIR) -> str:
    """Point Habitat-Sim at train/ and val/ basis meshes.

    The shipped ``hm3d_annotated_train_basis`` config lists scene folders
    without the ``train/`` prefix, so those paths do not resolve here.
    """
    path = scene_config_path(scenes_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "stages": {
            "paths": {
                ".glb": [
                    "train/*/*.basis.glb",
                    "val/*/*.basis.glb",
                ]
            }
        }
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return path


def build_habitat_config(
    split: str = "train",
    scenes: Optional[Sequence[str]] = None,
    gpu_device_id: int = 0,
    max_episode_steps: int = 500,
    episode_root: str = DEFAULT_EPISODE_ROOT,
    scenes_dir: str = DEFAULT_SCENES_DIR,
    seed: int = 12345,
):
    from habitat.config.default import get_config
    from habitat.config import read_write

    data_path = os.path.join(episode_root, "{split}", "{split}.json.gz")
    overrides: List[str] = [
        f"habitat.dataset.split={split}",
        f"habitat.dataset.scenes_dir={scenes_dir}",
        f"habitat.simulator.habitat_sim_v0.gpu_device_id={int(gpu_device_id)}",
        f"habitat.environment.max_episode_steps={int(max_episode_steps)}",
        f"habitat.seed={int(seed)}",
    ]
    if scenes:
        joined = ",".join(scenes)
        overrides.append(f"habitat.dataset.content_scenes=[{joined}]")
    config = get_config(
        "benchmark/nav/objectnav/objectnav_hm3d.yaml",
        overrides=overrides,
    )
    scene_cfg = ensure_scene_dataset_config(scenes_dir)
    with read_write(config):
        config.habitat.dataset.data_path = data_path
        config.habitat.simulator.scene_dataset = scene_cfg
    return config


def _keep_reference_replay() -> None:
    """Keep human-demo actions that ObjectNav-v1 would otherwise reject."""
    import habitat.datasets.object_nav.object_nav_dataset as dataset_mod

    if getattr(dataset_mod.ObjectGoalNavEpisode, "_keeps_replay", False):
        return
    original = dataset_mod.ObjectGoalNavEpisode

    def episode_with_replay(*args, **kwargs):
        replay = kwargs.pop("reference_replay", None)
        for key in ("attempts", "is_thda", "scene_dataset", "scene_state"):
            kwargs.pop(key, None)
        episode = original(*args, **kwargs)
        episode.reference_replay = replay
        return episode

    episode_with_replay._keeps_replay = True
    dataset_mod.ObjectGoalNavEpisode = episode_with_replay


def load_dataset(config, scene_config: Optional[str] = None):
    from habitat.datasets import make_dataset

    _keep_reference_replay()

    habitat_cfg = config.habitat if "habitat" in config else config
    dataset = make_dataset(habitat_cfg.dataset.type, config=habitat_cfg.dataset)
    if len(dataset.episodes) == 0:
        raise RuntimeError(
            "HM3D ObjectNav dataset has no episodes. "
            f"data_path={habitat_cfg.dataset.data_path} split={habitat_cfg.dataset.split}"
        )
    resolved = scene_config or ensure_scene_dataset_config(
        habitat_cfg.dataset.scenes_dir
    )
    for episode in dataset.episodes:
        episode.scene_dataset_config = resolved
    return dataset


class ObjectNavRLEnv:
    """Gym-style wrapper. Defined in make_env so workers can import it."""

    def __init__(self, config, dataset=None):
        from habitat.core.env import RLEnv

        class _Env(RLEnv):
            def get_reward_range(self):
                return (float("-inf"), float("inf"))

            def reset(self):
                observations = super().reset()
                self._prev_xz = self._agent_xz()
                return observations

            def _agent_xz(self):
                position = self._env.sim.get_agent_state().position
                return float(position[0]), float(position[2])

            def get_reward(self, observations):
                # RobustNav reward, with AllenAct's clip so a navmesh jump
                # cannot pay more than the distance the agent actually moved:
                # r_t = 10 * I_success + clip(d_{t-1} - d_t, ±moved) - 0.01.
                metrics = self._env.get_metrics()
                shaping = float(metrics.get("distance_to_goal_reward", 0.0))
                current = self._agent_xz()
                previous = getattr(self, "_prev_xz", None)
                if previous is not None:
                    moved = math.hypot(current[0] - previous[0], current[1] - previous[1])
                    shaping = max(-moved, min(moved, shaping))
                self._prev_xz = current
                success = float(metrics.get("success", 0.0))
                return (
                    shaping
                    + DEFAULT_CONFIG.step_penalty
                    + DEFAULT_CONFIG.success_reward * success
                )

            def get_done(self, observations):
                return bool(self._env.episode_over)

            def get_info(self, observations):
                return dict(self._env.get_metrics())

        self._env = _Env(config, dataset)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.original_action_space = self._env.action_space
        self.number_of_episodes = self._env.number_of_episodes

    @property
    def current_episode(self):
        return self._env.current_episode()

    @property
    def episodes(self):
        return self._env.episodes

    def seed(self, seed):
        self._env.seed(seed)

    def reset(self):
        return self._env.reset()

    def step(self, action):
        return self._env.step(action)

    def expert_action(self):
        """Shortest-path action for the current state, before ``step``."""
        from spikingnav.habitat_objectnav.expert import query_shortest_path_expert

        action, valid, follower = query_shortest_path_expert(
            self._env, getattr(self, "_follower", None)
        )
        self._follower = follower
        return int(action), bool(valid)

    def demo_action(self):
        """Next recorded human action for the current episode step."""
        from spikingnav.habitat_objectnav.expert import demo_action_index

        habitat_env = self._env.habitat_env
        episode = habitat_env.current_episode
        names = []
        index = 0
        while True:
            try:
                names.append(habitat_env.task.get_action_name(index))
            except (ValueError, IndexError):
                break
            index += 1
        action, valid = demo_action_index(
            getattr(episode, "reference_replay", None),
            int(getattr(habitat_env, "_elapsed_steps", 0)),
            names,
        )
        return int(action), bool(valid)

    def close(self):
        self._env.close()


def make_env_fn(config, scene_config: str, rank: int = 0):
    from habitat.config import read_write

    dataset = load_dataset(config, scene_config)
    with read_write(config):
        config.habitat.simulator.scene = dataset.episodes[0].scene_id
        config.habitat.simulator.scene_dataset = scene_config
    env = ObjectNavRLEnv(config, dataset)
    seed = int(config.habitat.seed) + int(rank)
    env.seed(seed)
    return env
