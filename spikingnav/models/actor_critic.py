"""Standalone and AllenAct actor-critic wrappers for SpikingNav."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import os

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from spikingnav.config import DEFAULT_CONFIG, SpikingNavConfig
from spikingnav.models.spn import SpikingPolicyNetwork
from spikingnav.models.sse import SpikingSensingEncoder


def _to_nchw(rgb: torch.Tensor) -> torch.Tensor:
    """Accept NHWC or NCHW RGB tensors."""
    if rgb.dim() == 3:
        rgb = rgb.unsqueeze(0)
    if rgb.shape[-1] in (1, 3, 4) and rgb.shape[1] not in (1, 3, 4):
        return rgb.permute(0, 3, 1, 2).contiguous()
    return rgb


class SpikingNavModel(nn.Module):
    """End-to-end SSE + SPN used for training / unit tests."""

    def __init__(
        self,
        action_dim: int,
        num_categories: Optional[int] = 12,
        cfg: Optional[SpikingNavConfig] = None,
        pretrained_ann: bool = True,
    ) -> None:
        super().__init__()
        self.cfg = cfg or DEFAULT_CONFIG
        self.action_dim = action_dim
        self.sse = SpikingSensingEncoder(
            cfg=self.cfg,
            num_categories=num_categories,
            pretrained_ann=pretrained_ann,
        )
        self.spn = SpikingPolicyNetwork(
            input_dim=self.sse.out_dim,
            action_dim=action_dim,
            cfg=self.cfg,
        )

    def initial_memory(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.cfg.hidden_size, device=device)

    def encode(self, rgb: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        rgb = _to_nchw(rgb)
        chunk = int(os.environ.get("SPIKINGNAV_VIS_CHUNK", "4"))
        batch = rgb.shape[0]
        if chunk <= 0 or batch <= chunk or not torch.is_grad_enabled():
            return self.sse(rgb, goal)
        parts = []
        for start in range(0, batch, chunk):
            end = min(start + chunk, batch)
            parts.append(
                checkpoint(
                    self.sse,
                    rgb[start:end],
                    goal[start:end],
                    use_reentrant=False,
                )
            )
        return torch.cat(parts, 0)

    def forward(
        self,
        rgb: torch.Tensor,
        goal: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (logits, values, new_memory).

        rgb: [T, N, H, W, C] / [N, H, W, C] / NCHW
        goal: [T, N, ...] or [N, ...]
        memory: [N, H]
        masks: [T, N, 1]
        """
        if rgb.dim() == 4:
            rgb = rgb.unsqueeze(0)
            if goal.dim() in (1, 2) and (masks is None or masks.shape[0] == 1):
                goal = goal.unsqueeze(0)
        t_steps, batch = rgb.shape[:2]
        device = rgb.device
        if memory is None:
            memory = self.initial_memory(batch, device)

        r_list = []
        for t in range(t_steps):
            goal_t = goal[t] if goal.dim() >= 2 and goal.shape[0] == t_steps else goal
            if goal_t.dim() > 1 and goal_t.shape[0] == 1 and batch > 1:
                goal_t = goal_t.expand(batch, *goal_t.shape[1:])
            r_list.append(self.encode(rgb[t], goal_t))
        r = torch.stack(r_list, 0)
        return self.spn(r, memory, masks)

    def act(
        self,
        rgb: torch.Tensor,
        goal: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values, memory = self.forward(rgb, goal, memory)
        logits = logits[-1]
        values = values[-1]
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.probs.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), values.squeeze(-1), memory


def flatten_leading(x: torch.Tensor, n_flat: int) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    lead = tuple(x.shape[:n_flat])
    return x.reshape(-1, *x.shape[n_flat:]), lead


class SpikingNavActorCritic:
    """Factory that builds an AllenAct ActorCriticModel when AllenAct is present."""

    @staticmethod
    def build(
        action_space,
        observation_space,
        goal_sensor_uuid: str,
        rgb_uuid: str = "rgb_lowres",
        num_categories: Optional[int] = 12,
        cfg: Optional[SpikingNavConfig] = None,
        pretrained_ann: bool = True,
    ):
        from allenact.algorithms.onpolicy_sync.policy import ActorCriticModel
        from allenact.base_abstractions.distributions import CategoricalDistr
        from allenact.base_abstractions.misc import ActorCriticOutput, Memory

        cfg = cfg or DEFAULT_CONFIG

        class _Model(ActorCriticModel[CategoricalDistr]):
            def __init__(self) -> None:
                super().__init__(
                    action_space=action_space, observation_space=observation_space
                )
                self.goal_sensor_uuid = goal_sensor_uuid
                self.rgb_uuid = rgb_uuid
                self.core = SpikingNavModel(
                    action_dim=action_space.n,
                    num_categories=num_categories,
                    cfg=cfg,
                    pretrained_ann=pretrained_ann,
                )
                self.memory_key = "membrane"

            @property
            def recurrent_hidden_state_size(self) -> int:
                return cfg.hidden_size

            @property
            def num_recurrent_layers(self) -> int:
                return 1

            def _recurrent_memory_specification(self):
                return {
                    self.memory_key: (
                        (
                            ("layer", 1),
                            ("sampler", None),
                            ("hidden", cfg.hidden_size),
                        ),
                        torch.float32,
                    )
                }

            def forward(  # type: ignore[override]
                self,
                observations: Dict[str, torch.Tensor],
                memory: Memory,
                prev_actions: torch.Tensor,
                masks: torch.FloatTensor,
            ):
                rgb = observations[self.rgb_uuid]
                goal = observations[self.goal_sensor_uuid]
                nsteps, nsamplers = masks.shape[:2]
                rgb_flat, _ = flatten_leading(rgb, 2)
                if goal.dim() >= 2 and goal.shape[0] == nsteps:
                    goal_flat, _ = flatten_leading(goal, 2)
                else:
                    goal_flat = goal.reshape(nsteps * nsamplers, *goal.shape[2:])

                r = self.core.encode(rgb_flat, goal_flat).view(nsteps, nsamplers, -1)
                u_prev = memory.tensor(self.memory_key)[0]
                logits, values, u_t = self.core.spn(r, u_prev, masks)
                mem = memory.set_tensor(self.memory_key, u_t.unsqueeze(0))
                return (
                    ActorCriticOutput(
                        distributions=CategoricalDistr(logits=logits),
                        values=values,
                        extras={},
                    ),
                    mem,
                )

        return _Model()
