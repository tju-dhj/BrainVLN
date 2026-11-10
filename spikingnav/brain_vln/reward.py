"""Discrete-action reward used by MLANet's released PPO config.

MLANet's ``VLNCERLEnv`` and ``mlanet/config/mla/mla_ppo.yaml`` use the change
in geodesic distance, a flat slack, and a success bonus:

    r = slack + distance_scalar * (d_prev - d) + success_reward * success

The released PPO file sets:

    slack_reward = -0.02
    distance_scalar = 1.0
    success_reward = 8.0

Integer actions use the flat slack. A non-finite distance change is replaced
by -1 before the scalar, because this simulator path can return a non-finite
geodesic. Habitat's own distance measure usually stays finite.

Success follows the VLN-CE ``Success`` measure: STOP has been called and the
geodesic distance is at most 3 m. Standing inside 3 m without stopping does
not pay the bonus.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MLANetReward:
    slack_reward: float = -0.02
    success_reward: float = 8.0
    distance_scalar: float = 1.0
    success_distance: float = 3.0

    def __post_init__(self) -> None:
        if self.slack_reward > 0:
            raise ValueError("slack_reward must be <= 0")
        if self.success_reward < 0:
            raise ValueError("success_reward must be >= 0")
        if self.distance_scalar < 0:
            raise ValueError("distance_scalar must be >= 0")
        if self.success_distance <= 0:
            raise ValueError("success_distance must be positive")


def goal_success(stopped: bool, distance_to_goal: float, success_distance: float = 3.0) -> bool:
    """VLN-CE success: the agent called STOP inside the success radius."""
    if not stopped or not math.isfinite(distance_to_goal):
        return False
    return float(distance_to_goal) <= float(success_distance)


def step_reward(
    previous_distance: float,
    distance: float,
    stopped: bool,
    reward: MLANetReward = MLANetReward(),
) -> float:
    """One environment step of the MLANet / VLN-CE discrete reward."""
    delta = float(previous_distance) - float(distance)
    if not math.isfinite(delta):
        delta = -1.0
    value = float(reward.slack_reward)
    value += float(reward.distance_scalar) * delta
    if goal_success(stopped, distance, reward.success_distance):
        value += float(reward.success_reward)
    return value
