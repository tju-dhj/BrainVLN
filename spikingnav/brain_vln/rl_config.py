"""Turn configs/brain_vln_rl.yaml into the objects the PPO trainer uses."""

from __future__ import annotations

from types import SimpleNamespace

from omegaconf import OmegaConf

from spikingnav.brain_vln.ppo import PPOConfig
from spikingnav.brain_vln.reward import MLANetReward

_TOP = {
    "init",
    "eval_root",
    "output",
    "data",
    "scenes",
    "device",
    "sim_gpu",
    "seed",
    "environments",
    "updates",
    "rollout_steps",
    "mini_batches",
    "checkpoints",
    "reward_window",
    "resume_interval",
    "max_episode_steps",
    "ppo",
    "reward",
}
_PPO = {
    "clip",
    "epochs",
    "value_coef",
    "entropy_coef",
    "learning_rate",
    "max_grad_norm",
    "gamma",
    "gae_lambda",
    "adam_eps",
    "normalize_advantage",
    "linear_lr_decay",
    "linear_clip_decay",
}
_REWARD = {"slack_reward", "success_reward", "distance_scalar", "success_distance"}
_POSITIVE = (
    "environments",
    "updates",
    "rollout_steps",
    "mini_batches",
    "checkpoints",
    "reward_window",
    "resume_interval",
    "max_episode_steps",
)


def settings_from_config(cfg) -> SimpleNamespace:
    """Validate a Hydra config and build the PPO and reward objects."""
    values = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(values, dict):
        raise TypeError("RL config must be a mapping")
    values.pop("hydra", None)
    _exact_keys(values, _TOP, "config")
    _exact_keys(values["ppo"], _PPO, "ppo")
    _exact_keys(values["reward"], _REWARD, "reward")
    for key in ("output", "data", "scenes", "eval_root", "device"):
        if not str(values[key]).strip():
            raise ValueError(f"{key} must be set in configs/brain_vln_rl.yaml")
    for key in _POSITIVE:
        if int(values[key]) < 1:
            raise ValueError(f"{key} must be positive")
    ppo_values = dict(values["ppo"])
    decay_lr = bool(ppo_values.pop("linear_lr_decay"))
    decay_clip = bool(ppo_values.pop("linear_clip_decay"))
    reward_values = dict(values["reward"])
    init = str(values["init"]).strip()
    return SimpleNamespace(
        init=init or None,
        eval_root=str(values["eval_root"]),
        output=str(values["output"]),
        data=str(values["data"]),
        scenes=str(values["scenes"]),
        device=str(values["device"]),
        sim_gpu=int(values["sim_gpu"]),
        seed=int(values["seed"]),
        environments=int(values["environments"]),
        updates=int(values["updates"]),
        rollout_steps=int(values["rollout_steps"]),
        mini_batches=int(values["mini_batches"]),
        checkpoints=int(values["checkpoints"]),
        reward_window=int(values["reward_window"]),
        resume_interval=int(values["resume_interval"]),
        max_episode_steps=int(values["max_episode_steps"]),
        linear_lr_decay=decay_lr,
        linear_clip_decay=decay_clip,
        ppo=PPOConfig(**ppo_values),
        reward=MLANetReward(**reward_values),
    )


def _exact_keys(values: dict, expected: set, label: str) -> None:
    if not isinstance(values, dict):
        raise TypeError(f"{label} must be a mapping")
    missing = sorted(expected - set(values))
    unknown = sorted(set(values) - expected)
    if missing or unknown:
        parts = []
        if missing:
            parts.append("missing " + ", ".join(missing))
        if unknown:
            parts.append("unknown " + ", ".join(unknown))
        raise ValueError(f"{label}: " + "; ".join(parts))
