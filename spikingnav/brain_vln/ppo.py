"""Clipped PPO fine-tune after offline imitation.

The defaults are the PPO block in MLANet's ``mla_ppo.yaml``: clip 0.1, one
epoch, learning rate 2.5e-5, gradient norm 0.5, and linear decay of both the
learning rate and the clip range. Advantages are not normalized. The critic
reads the same Action and Goal membrane the action head reads. Encoders stay
frozen.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn as nn
from torch.distributions import Categorical

from spikingnav.brain_vln.model import BrainVLNModel
from spikingnav.brain_vln.state import ReservoirState


@dataclass(frozen=True)
class PPOConfig:
    """MLANet ``mla_ppo.yaml`` PPO block. Advantages stay unnormalized."""

    clip: float = 0.1
    epochs: int = 1
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 2.5e-5
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    adam_eps: float = 1.0e-5
    normalize_advantage: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.clip < 1.0:
            raise ValueError("clip must lie in (0, 1)")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.gamma < 0.0 or self.gae_lambda < 0.0:
            raise ValueError("gamma and gae_lambda must be >= 0")

    def decayed(self, update: int, updates: int) -> "PPOConfig":
        """Linear learning-rate and clip decay used by MLANet's PPO trainer.

        Update ``k`` runs at scale ``1 - k / updates``, matching Habitat's
        ``percent_done`` before the update counter increments.
        """
        return self.scheduled(update, updates, decay_lr=True, decay_clip=True)

    def scheduled(self, update: int, updates: int, *, decay_lr: bool, decay_clip: bool) -> "PPOConfig":
        """Scale the learning rate and clip for this update. Either decay can stay off."""
        if updates < 1 or not 0 <= update < updates:
            raise ValueError("update must lie in [0, updates)")
        scale = 1.0 - (float(update) / float(updates))
        return replace(
            self,
            clip=self.clip * scale if decay_clip else self.clip,
            learning_rate=self.learning_rate * scale if decay_lr else self.learning_rate,
        )


class ValueHead(nn.Module):
    """Scalar critic on the Action and Goal membrane."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        if in_dim < 1:
            raise ValueError("value features must be non-empty")
        self.proj = nn.Linear(in_dim, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features).squeeze(-1)


def value_features(model: BrainVLNModel, state: ReservoirState) -> torch.Tensor:
    bounds = model.bounds
    action = state.membrane[:, bounds["action"][0] : bounds["action"][1]]
    goal = state.membrane[:, bounds["goal"][0] : bounds["goal"][1]]
    return torch.cat((action, goal), dim=-1)


def generalized_advantage(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    bootstrap: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple:
    """GAE-Lambda. ``dones[t]`` is true when the episode ends at step ``t``.

    ``rewards`` and ``values`` have shape ``[T]`` or ``[T, B]``. ``bootstrap``
    is the value after the last step and is ignored where the last step ended
    an episode. A finished episode does not bootstrap from the next stored value.
    """
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError("rewards, values, and dones must share a shape")
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(bootstrap)
    for step in reversed(range(rewards.shape[0])):
        if step + 1 < rewards.shape[0]:
            next_value = values[step + 1]
        else:
            next_value = bootstrap
        mask = 1.0 - dones[step].to(dtype=rewards.dtype)
        next_value = next_value * mask
        delta = rewards[step] + gamma * next_value - values[step]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[step] = gae
    return advantages, advantages + values


def ppo_loss(
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantage: torch.Tensor,
    value: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
    config: PPOConfig = PPOConfig(),
) -> tuple:
    """Clipped policy loss, value loss, and entropy bonus. Returns the sum and parts."""
    ratio = torch.exp(log_prob - old_log_prob)
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - config.clip, 1.0 + config.clip) * advantage
    policy = -torch.min(unclipped, clipped).mean()
    value_loss = 0.5 * (value - returns).pow(2).mean()
    entropy_loss = -entropy.mean()
    total = policy + config.value_coef * value_loss + config.entropy_coef * entropy_loss
    return total, {
        "policy": float(policy.detach()),
        "value": float(value_loss.detach()),
        "entropy": float((-entropy_loss).detach()),
    }


def distribution(logits: torch.Tensor) -> Categorical:
    return Categorical(logits=logits)


def evaluate_action(logits: torch.Tensor, action: torch.Tensor) -> tuple:
    dist = distribution(logits)
    return dist.log_prob(action), dist.entropy()


_STATE_FIELDS = (
    "membrane",
    "spike",
    "adaptation",
    "synaptic_current",
    "delay_queue",
    "time",
    "spatial_trace",
    "staleness",
    "hidden",
    "pre_trace",
    "post_trace",
    "stop_membrane",
)


def move_state(state: ReservoirState, device: torch.device) -> ReservoirState:
    kwargs = {}
    for name in _STATE_FIELDS:
        value = getattr(state, name)
        kwargs[name] = value if value is None else value.to(device)
    return ReservoirState(**kwargs)


@dataclass
class RolloutStep:
    """One sampled action and the reservoir state from before that step."""

    rgb: torch.Tensor
    depth: torch.Tensor
    depth_valid: torch.Tensor
    tokens: torch.Tensor
    token_mask: torch.Tensor
    instruction_text: str
    prev_action: int
    action: int
    old_log_prob: float
    value: float
    reward: float
    done: bool
    reset: bool
    state: ReservoirState


class RecurrentActor:
    """Sample discrete actions while carrying the reservoir across steps.

    The forward that chooses the action also produces the value. A rollout
    chunk that stops before the episode ends leaves that next observation
    cached, so the following chunk does not encode it twice.
    """

    def __init__(self, model: BrainVLNModel, critic: ValueHead) -> None:
        self.model = model
        self.critic = critic
        self.state = None
        self.prev_action = 4
        self.cache = None

    def reset(self) -> None:
        self.state = None
        self.prev_action = 4
        self.cache = None

    @torch.no_grad()
    def observe(self, observation: dict, tokens: torch.Tensor, token_mask: torch.Tensor, instruction_text: str = "") -> dict:
        if self.cache is not None:
            return self.cache
        device = next(self.model.parameters()).device
        reset = self.state is None
        if reset:
            state = self.model.initial_state(1, device)
        else:
            state = move_state(self.state, device)
        stored_state = move_state(state.detach(), torch.device("cpu"))
        rgb = observation["rgb"].to(device)
        depth = observation["depth"].to(device)
        depth_valid = observation["depth_valid"].to(device)
        token_batch = tokens.to(device).view(1, -1)
        mask_batch = token_mask.to(device).view(1, -1)
        valid = torch.ones(1, 1, dtype=torch.bool, device=device)
        reset_mask = torch.tensor([[reset]], dtype=torch.bool, device=device)
        detach = torch.zeros_like(valid)
        text = [instruction_text] if instruction_text else None
        logits, new_state, _aux = self.model(
            rgb.view(1, 1, *rgb.shape),
            depth.view(1, 1, *depth.shape),
            depth_valid.view(1, 1, *depth_valid.shape),
            token_batch,
            mask_batch,
            valid,
            reset_mask,
            detach,
            prev_action=torch.tensor([self.prev_action], dtype=torch.long, device=device),
            state=state,
            instruction_text=text,
        )
        value = self.critic(value_features(self.model, new_state))
        self.cache = {
            "logits": logits[-1, 0].detach(),
            "value": float(value.reshape(-1)[0].detach()),
            "state": stored_state,
            "new_state": new_state.detach(),
            "reset": reset,
            "prev_action": int(self.prev_action),
            "rgb": observation["rgb"].detach().cpu(),
            "depth": observation["depth"].detach().cpu(),
            "depth_valid": observation["depth_valid"].detach().cpu(),
            "tokens": tokens.detach().cpu().view(-1),
            "token_mask": token_mask.detach().cpu().view(-1),
            "instruction_text": instruction_text or "",
        }
        return self.cache

    def sample(self) -> tuple:
        if self.cache is None:
            raise RuntimeError("observe an image before sampling an action")
        dist = distribution(self.cache["logits"])
        action = dist.sample()
        return int(action.item()), float(dist.log_prob(action).item())

    def commit(self, action: int) -> None:
        if self.cache is None:
            raise RuntimeError("there is no sampled step to commit")
        self.state = self.cache["new_state"]
        self.prev_action = int(action)
        self.cache = None


def collect_segment(actor: RecurrentActor, env, tokens, token_mask, instruction_text: str, rollout_steps: int) -> tuple:
    """Sample up to ``rollout_steps`` actions, stopping early when the episode ends."""
    if rollout_steps < 1:
        raise ValueError("rollout_steps must be positive")
    if actor.cache is None:
        actor.observe(env.observation(), tokens, token_mask, instruction_text)
    steps = []
    for _ in range(rollout_steps):
        action, log_prob = actor.sample()
        cached = actor.cache
        _obs, reward, done, _info = env.step(action)
        steps.append(
            RolloutStep(
                rgb=cached["rgb"],
                depth=cached["depth"],
                depth_valid=cached["depth_valid"],
                tokens=cached["tokens"],
                token_mask=cached["token_mask"],
                instruction_text=cached["instruction_text"],
                prev_action=cached["prev_action"],
                action=int(action),
                old_log_prob=float(log_prob),
                value=float(cached["value"]),
                reward=float(reward),
                done=bool(done),
                reset=bool(cached["reset"]),
                state=cached["state"],
            )
        )
        actor.commit(action)
        if done:
            break
        actor.observe(env.observation(), tokens, token_mask, instruction_text)
    bootstrap = 0.0 if steps[-1].done else float(actor.cache["value"])
    return steps, bootstrap


def advantages_for_segments(segments: list, gamma: float, gae_lambda: float) -> tuple:
    """GAE for each environment segment, then concatenate. Boundaries do not leak."""
    if not segments:
        raise ValueError("PPO update needs at least one segment")
    packed = []
    advantages = []
    returns = []
    rewards = []
    for steps, bootstrap in segments:
        if not steps:
            raise ValueError("PPO update needs at least one step")
        reward = torch.tensor([step.reward for step in steps], dtype=torch.float32)
        value = torch.tensor([step.value for step in steps], dtype=torch.float32)
        done = torch.tensor([step.done for step in steps], dtype=torch.bool)
        advantage, outcome = generalized_advantage(
            reward,
            value,
            done,
            torch.tensor(float(bootstrap), dtype=torch.float32),
            gamma,
            gae_lambda,
        )
        packed.extend(steps)
        advantages.append(advantage.detach())
        returns.append(outcome.detach())
        rewards.append(reward)
    return packed, torch.cat(advantages), torch.cat(returns), torch.cat(rewards)


def ppo_update(
    model: BrainVLNModel,
    critic: ValueHead,
    optimizer,
    steps: list,
    bootstrap: float,
    config: PPOConfig = PPOConfig(),
    minibatch_size: int = 4,
    segments: list | None = None,
) -> dict:
    """One clipped PPO pass.

    Advantages stay raw unless ``config.normalize_advantage`` is set. ``segments``
    is a list of ``(steps, bootstrap)`` pairs, one per environment. A single
    ``steps`` list remains the one-environment call.
    """
    if segments is None:
        if not steps:
            raise ValueError("PPO update needs at least one step")
        segments = [(steps, bootstrap)]
    if minibatch_size < 1:
        raise ValueError("minibatch_size must be positive")
    packed, advantages, returns, rewards = advantages_for_segments(segments, config.gamma, config.gae_lambda)
    if config.normalize_advantage and advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-8)
    model.train()
    critic.train()
    totals = []
    parts = []
    count = len(packed)
    for _epoch in range(config.epochs):
        order = torch.randperm(count)
        for start in range(0, count, minibatch_size):
            chosen = order[start : start + minibatch_size]
            optimizer.zero_grad(set_to_none=True)
            loss, report = _minibatch_loss(model, critic, packed, chosen, advantages, returns, config)
            loss.backward()
            trainable = [param for param in list(model.parameters()) + list(critic.parameters()) if param.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
            optimizer.step()
            totals.append(float(loss.detach()))
            parts.append(report)
    return {
        "loss": sum(totals) / len(totals),
        "policy": sum(item["policy"] for item in parts) / len(parts),
        "value": sum(item["value"] for item in parts) / len(parts),
        "entropy": sum(item["entropy"] for item in parts) / len(parts),
        "mean_reward": float(rewards.mean()),
        "steps": count,
    }


def _minibatch_loss(model, critic, steps, index, advantages, returns, config: PPOConfig):
    chosen = [steps[int(item)] for item in index]
    device = next(model.parameters()).device
    state = ReservoirState.merge([move_state(step.state, device) for step in chosen])
    rgb = torch.stack([step.rgb for step in chosen], dim=0).unsqueeze(0).to(device)
    depth = torch.stack([step.depth for step in chosen], dim=0).unsqueeze(0).to(device)
    depth_valid = torch.stack([step.depth_valid for step in chosen], dim=0).unsqueeze(0).to(device)
    tokens = torch.stack([step.tokens for step in chosen], dim=0).to(device)
    token_mask = torch.stack([step.token_mask for step in chosen], dim=0).to(device)
    batch = len(chosen)
    valid = torch.ones(1, batch, dtype=torch.bool, device=device)
    reset = torch.tensor([[step.reset for step in chosen]], dtype=torch.bool, device=device)
    detach = torch.zeros_like(valid)
    prev_action = torch.tensor([step.prev_action for step in chosen], dtype=torch.long, device=device)
    texts = [step.instruction_text for step in chosen]
    instruction = texts if any(texts) else None
    logits, new_state, _aux = model(
        rgb,
        depth,
        depth_valid,
        tokens,
        token_mask,
        valid,
        reset,
        detach,
        prev_action=prev_action,
        state=state,
        instruction_text=instruction,
    )
    action = torch.tensor([step.action for step in chosen], dtype=torch.long, device=device)
    log_prob, entropy = evaluate_action(logits[-1], action)
    value = critic(value_features(model, new_state))
    old_log_prob = torch.tensor([step.old_log_prob for step in chosen], dtype=torch.float32, device=device)
    chosen_index = index.to(device)
    return ppo_loss(
        log_prob,
        old_log_prob,
        advantages.to(device)[chosen_index],
        value,
        returns.to(device)[chosen_index],
        entropy,
        config,
    )
