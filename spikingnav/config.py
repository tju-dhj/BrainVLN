"""Default hyperparameters inferred from the paper and RobustNav / AllenAct."""

from dataclasses import dataclass, field
from typing import List, Tuple


OBJECTNAV_TARGETS: Tuple[str, ...] = (
    "AlarmClock",
    "Apple",
    "BaseballBat",
    "BasketBall",
    "Bowl",
    "GarbageCan",
    "HousePlant",
    "Laptop",
    "Mug",
    "SprayBottle",
    "Television",
    "Vase",
)

VISUAL_CORRUPTIONS: Tuple[str, ...] = (
    "Low Lighting",
    "Motion Blur",
    "Camera Crack",
    "Defocus Blur",
    "Speckle Noise",
    "Lower FOV",
    "Spatter",
)


@dataclass
class SpikingNavConfig:
    # Visual / SSE
    image_size: int = 224
    time_steps: int = 4
    visual_channels: int = 32
    compressor_hidden: int = 128
    goal_dims: int = 32
    feature_hw: int = 7

    # SPN
    hidden_size: int = 512
    leak: float = 0.5
    threshold: float = 1.0
    surrogate_alpha: float = 2.0

    # PPO / AllenAct. SpikingNav adopts RobustNav's DD-PPO settings
    # (clip 0.1, value 0.5, entropy 0.01, Adam 3e-4 linear decay, γ 0.99,
    # GAE 0.95, rollout 128, 4 epochs, grad clip 0.5).
    # Reward from RobustNav S1, with AllenAct's motion clip on the geodesic term:
    # r_t = 10 * I_success + clip(d_{t-1} - d_t, ±distance moved) - 0.01.
    success_reward: float = 10.0
    step_penalty: float = -0.01
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    clip_param: float = 0.1
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    rollout_steps: int = 128
    update_repeats: int = 4
    max_grad_norm: float = 0.5
    pointnav_steps: int = 75_000_000
    objectnav_steps: int = 300_000_000

    # Environment
    camera_width: int = 400
    camera_height: int = 300
    step_size: float = 0.25
    rotation_degrees: float = 30.0
    max_steps: int = 500
    horizontal_fov: float = 79.0
    lower_fov: float = 40.0
    distance_to_goal: float = 0.2
    visibility_distance: float = 1.0
    target_types: Tuple[str, ...] = field(default_factory=lambda: OBJECTNAV_TARGETS)


DEFAULT_CONFIG = SpikingNavConfig()
