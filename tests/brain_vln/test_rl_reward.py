"""MLANet PPO reward, checkpoint selection, and the clipped PPO update."""

import json
import math
from types import SimpleNamespace

import torch

from spikingnav.brain_vln.config import tiny_config
from spikingnav.brain_vln.il_checkpoint import checkpoint_due, select_best_il_checkpoint, shard_episodes
from spikingnav.brain_vln.model import BrainVLNModel
from spikingnav.brain_vln.ppo import (
    PPOConfig,
    RecurrentActor,
    ValueHead,
    advantages_for_segments,
    collect_segment,
    generalized_advantage,
    ppo_loss,
    ppo_update,
)
from spikingnav.brain_vln.reward import MLANetReward, goal_success, step_reward


def test_flat_slack_is_not_scaled_by_distance():
    reward = step_reward(5.0, 4.0, stopped=False)
    assert math.isclose(reward, -0.02 + 1.0)
    assert not math.isclose(reward, -0.02 * (1.0 / 0.25) + 1.0)


def test_success_bonus_requires_stop_inside_three_meters():
    assert goal_success(True, 3.0) is True
    assert goal_success(True, 3.01) is False
    assert goal_success(False, 0.0) is False
    assert goal_success(True, float("inf")) is False
    stopped = step_reward(2.9, 2.9, stopped=True)
    walking = step_reward(2.9, 2.9, stopped=False)
    assert math.isclose(stopped, -0.02 + 8.0)
    assert math.isclose(walking, -0.02)


def test_nonfinite_distance_change_becomes_minus_one():
    reward = MLANetReward()
    assert step_reward(float("inf"), 1.0, False, reward) == reward.slack_reward + reward.distance_scalar * -1.0
    assert step_reward(float("inf"), float("inf"), False, reward) == reward.slack_reward - reward.distance_scalar
    assert step_reward(float("nan"), 1.0, False, reward) == reward.slack_reward - reward.distance_scalar


def test_gae_stops_at_episode_end_and_bootstraps_otherwise():
    rewards = torch.tensor([1.0, 0.0])
    values = torch.tensor([0.0, 0.0])
    dones = torch.tensor([False, True])
    advantages, returns = generalized_advantage(rewards, values, dones, torch.tensor(5.0), gamma=1.0, gae_lambda=1.0)
    assert torch.equal(advantages, torch.tensor([1.0, 0.0]))
    assert torch.equal(returns, advantages + values)
    one = generalized_advantage(
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([False]),
        torch.tensor(1.0),
        gamma=0.5,
        gae_lambda=1.0,
    )[0]
    assert torch.allclose(one, torch.tensor([0.5]))


def test_defaults_match_the_released_mlanet_ppo_file():
    config = PPOConfig()
    reward = MLANetReward()
    assert config.clip == 0.1
    assert config.epochs == 1
    assert config.learning_rate == 2.5e-5
    assert config.max_grad_norm == 0.5
    assert config.value_coef == 0.5
    assert config.entropy_coef == 0.01
    assert config.gamma == 0.99
    assert config.gae_lambda == 0.95
    assert reward.slack_reward == -0.02
    assert reward.success_reward == 8.0
    assert reward.distance_scalar == 1.0
    assert reward.success_distance == 3.0
    first = config.decayed(0, 100_000)
    assert first.clip == config.clip
    assert first.learning_rate == config.learning_rate
    later = config.decayed(50_000, 100_000)
    assert math.isclose(later.clip, 0.05)
    assert math.isclose(later.learning_rate, 1.25e-5)


def test_gae_does_not_cross_environment_boundary():
    def step(reward, value, done):
        return SimpleNamespace(reward=reward, value=value, done=done)

    _steps, advantages, _returns, _rewards = advantages_for_segments(
        [
            ([step(0.0, 0.0, False)], 1.0),
            ([step(0.0, 0.0, False)], 0.0),
        ],
        gamma=1.0,
        gae_lambda=1.0,
    )
    assert torch.equal(advantages, torch.tensor([1.0, 0.0]))


def test_finished_val_unseen_spl_selects_the_checkpoint(tmp_path):
    first = tmp_path / "update_001000"
    second = tmp_path / "update_002000"
    first.mkdir()
    second.mkdir()
    early = tmp_path / "early.pt"
    late = tmp_path / "late.pt"
    early.write_bytes(b"early")
    late.write_bytes(b"late")
    (first / "summary.json").write_text(
        json.dumps({"spl": 0.21, "sr": 0.23, "update": 1000, "checkpoint": str(early)}),
        encoding="utf-8",
    )
    (second / "summary.json").write_text(
        json.dumps({"spl": 0.35, "sr": 0.38, "update": 2000, "checkpoint": str(late)}),
        encoding="utf-8",
    )
    (tmp_path / "update_003000").mkdir()
    (tmp_path / "update_003000" / "episodes.jsonl").write_text("{}\n", encoding="utf-8")
    chosen, summary = select_best_il_checkpoint(tmp_path)
    assert chosen == late
    assert summary["spl"] == 0.35


def test_twenty_checkpoints_span_the_mlanet_update_budget():
    due = [update for update in range(100_000) if checkpoint_due(update, 100_000, 20)]
    assert len(due) == 20
    assert due[0] == 4999
    assert due[-1] == 99999


def test_scene_shards_cover_every_episode_once():
    episodes = [f"scene_{index // 3}/ep_{index}" for index in range(12)]
    shards = shard_episodes(episodes, 8)
    assert len(shards) == 8
    assert all(shards)
    flat = [episode for shard in shards for episode in shard]
    assert sorted(flat) == sorted(episodes)


def test_clip_keeps_the_pessimistic_surrogate():
    config = PPOConfig(clip=0.2, value_coef=0.0, entropy_coef=0.0)
    optimistic = ppo_loss(
        torch.tensor([math.log(2.0)]),
        torch.zeros(1),
        torch.ones(1),
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(1),
        config,
    )[0]
    pessimistic = ppo_loss(
        torch.tensor([math.log(2.0)]),
        torch.zeros(1),
        -torch.ones(1),
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(1),
        config,
    )[0]
    assert torch.allclose(optimistic, torch.tensor(-1.2))
    assert torch.allclose(pessimistic, torch.tensor(2.0))


class _FakeEnv:
    def __init__(self):
        self.calls = 0
        self._obs = {
            "rgb": torch.rand(3, 16, 16),
            "depth": torch.rand(1, 16, 16),
            "depth_valid": torch.ones(1, 16, 16, dtype=torch.bool),
        }

    def observation(self):
        return self._obs

    def step(self, action):
        self.calls += 1
        self._obs = {
            "rgb": torch.rand(3, 16, 16),
            "depth": torch.rand(1, 16, 16),
            "depth_valid": torch.ones(1, 16, 16, dtype=torch.bool),
        }
        done = self.calls == 2
        return self._obs, 0.2 if action == 1 else -0.05, done, {}


def test_ppo_update_on_a_tiny_rollout_is_finite():
    torch.manual_seed(0)
    model = BrainVLNModel(tiny_config())
    critic = ValueHead(model.config.populations.action + model.config.populations.goal)
    actor = RecurrentActor(model, critic)
    tokens = torch.ones(4, dtype=torch.long)
    mask = torch.ones(4, dtype=torch.bool)
    steps, bootstrap = collect_segment(actor, _FakeEnv(), tokens, mask, "", 4)
    assert len(steps) == 2
    assert steps[-1].done
    assert bootstrap == 0.0
    assert steps[0].reset is True
    assert steps[1].reset is False
    optimizer = torch.optim.Adam(
        [param for param in list(model.parameters()) + list(critic.parameters()) if param.requires_grad],
        lr=2.0e-4,
    )
    stats = ppo_update(model, critic, optimizer, steps, bootstrap, PPOConfig(epochs=1), minibatch_size=1)
    assert math.isfinite(stats["loss"])
    assert math.isfinite(stats["policy"])
    assert stats["steps"] == 2
