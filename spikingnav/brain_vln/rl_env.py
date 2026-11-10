"""Low-level R2R-CE episode for PPO. The goal is used only by the reward.

The policy sees RGB, depth, the instruction, and the previous action. Geodesic
distance is computed after the simulator steps and is not written into the
observation that the network reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from spikingnav.brain_vln.reward import MLANetReward, goal_success, step_reward
from spikingnav.vlnce.constants import MAX_EPISODE_STEPS, SIM_ACTIONS, STOP


class LowLevelEnv:
    """One Habitat agent, discrete actions, and the MLANet / VLN-CE reward."""

    def __init__(self, sim, goal, reward: MLANetReward = MLANetReward(), max_steps: int = MAX_EPISODE_STEPS) -> None:
        self.sim = sim
        self.goal = np.asarray(goal, dtype=np.float32)
        self.reward = reward
        self.max_steps = int(max_steps)
        self.distance = None
        self.start_distance = None
        self.steps = 0
        self.positions = []
        self._observation = None

    def reset(self, position, rotation) -> dict:
        from spikingnav.vlnce.sim import _place, geodesic

        _place(self.sim, position, rotation)
        self.steps = 0
        self.positions = []
        self.distance = float(geodesic(self.sim, _position(self.sim), self.goal))
        self.start_distance = self.distance
        self._observation = _sense(self.sim)
        self.positions.append(_position(self.sim))
        return self._observation

    def observation(self) -> dict:
        if self._observation is None:
            raise RuntimeError("reset the environment before reading an observation")
        return self._observation

    def step(self, action: int) -> tuple:
        from spikingnav.vlnce.sim import geodesic

        action = int(action)
        if action not in (STOP, *SIM_ACTIONS):
            raise ValueError(f"unknown action {action}")
        stopped = action == STOP
        if not stopped:
            self.sim.step(SIM_ACTIONS[action])
        self.steps += 1
        distance = float(geodesic(self.sim, _position(self.sim), self.goal))
        reward = step_reward(self.distance, distance, stopped, self.reward)
        self.distance = distance
        self._observation = _sense(self.sim)
        self.positions.append(_position(self.sim))
        done = stopped or self.steps >= self.max_steps
        success = goal_success(stopped, distance, self.reward.success_distance)
        info = {
            "distance": distance,
            "success": success,
            "stopped": stopped,
            "steps": self.steps,
            "start_distance": self.start_distance,
        }
        return self._observation, reward, done, info


def _position(sim):
    return np.asarray(sim.get_agent(0).get_state().position, dtype=np.float32).copy()


def _sense(sim) -> dict:
    from spikingnav.vlnce.sim import _depth, _rgb

    rgb = np.asarray(_rgb(sim))
    depth = np.asarray(_depth(sim), dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    valid = np.isfinite(depth) & (depth > 0)
    depth = np.where(valid, depth, 0.0).astype(np.float32)
    return {
        "rgb": torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div(255.0),
        "depth": torch.from_numpy(np.ascontiguousarray(depth)).unsqueeze(0),
        "depth_valid": torch.from_numpy(np.ascontiguousarray(valid)).unsqueeze(0),
    }


def load_episode_task(episode_dir: Path) -> dict:
    """Pose and goal for the reward, plus the instruction the policy is allowed to see."""
    from spikingnav.brain_vln.data.episode import read_header

    episode_dir = Path(episode_dir)
    header = read_header(episode_dir)
    metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
    trajectory = json.loads((episode_dir / "trajectories" / "trajectory.json").read_text(encoding="utf-8"))
    reference = trajectory.get("reference_path") or []
    goal = metadata.get("goal")
    position = metadata.get("start_position")
    rotation = metadata.get("start_rotation")
    if not isinstance(goal, list) or len(goal) != 3:
        raise ValueError(f"{episode_dir.name} has no goal position")
    if not isinstance(position, list) or len(position) != 3:
        raise ValueError(f"{episode_dir.name} has no start position")
    if not isinstance(rotation, list) or len(rotation) != 4:
        raise ValueError(f"{episode_dir.name} has no start rotation")
    return {
        "episode_dir": episode_dir,
        "episode_id": header.episode_id,
        "scene": header.scene,
        "scene_id": header.scene_id,
        "instruction_text": header.instruction_text,
        "tokens": header.tokens,
        "token_mask": header.token_mask,
        "start_position": [float(value) for value in position],
        "start_rotation": [float(value) for value in rotation],
        "goal": [float(value) for value in goal],
        "reference_path": reference,
    }
