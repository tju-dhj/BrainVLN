"""Hydra config for BrainVLN PPO."""

from pathlib import Path

import pytest

hydra = pytest.importorskip("hydra")
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from spikingnav.brain_vln.rl_config import settings_from_config

ROOT = Path(__file__).resolve().parents[2]


def setup_function():
    GlobalHydra.instance().clear()


def teardown_function():
    GlobalHydra.instance().clear()


def _compose(overrides=None):
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        return compose(config_name="brain_vln_rl", overrides=overrides or [])


def test_rl_config_uses_the_mlanet_ppo_defaults():
    args = settings_from_config(_compose())
    assert args.init is None
    assert args.environments == 8
    assert args.updates == 100_000
    assert args.rollout_steps == 32
    assert args.mini_batches == 1
    assert args.checkpoints == 20
    assert args.reward_window == 50
    assert args.resume_interval == 100
    assert args.max_episode_steps == 500
    assert args.ppo.clip == 0.1
    assert args.ppo.epochs == 1
    assert args.ppo.learning_rate == 2.5e-5
    assert args.ppo.max_grad_norm == 0.5
    assert args.linear_lr_decay is True
    assert args.linear_clip_decay is True
    assert args.ppo.normalize_advantage is False
    assert args.reward.slack_reward == -0.02
    assert args.reward.success_reward == 8.0
    assert args.reward.distance_scalar == 1.0
    assert args.reward.success_distance == 3.0


def test_rl_config_accepts_a_command_override():
    args = settings_from_config(_compose(["environments=4", "ppo.clip=0.2", "reward.slack_reward=-0.01"]))
    assert args.environments == 4
    assert args.ppo.clip == 0.2
    assert args.reward.slack_reward == -0.01


def test_rl_config_rejects_an_unknown_key():
    cfg = OmegaConf.create(OmegaConf.to_container(_compose(), resolve=True))
    cfg.extra = 1
    with pytest.raises(ValueError, match="unknown extra"):
        settings_from_config(cfg)
