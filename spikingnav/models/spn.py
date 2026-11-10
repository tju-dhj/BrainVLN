"""Spiking Policy Network (paper Sec. III-C2, Eqs. 7-13)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from spikingnav.config import DEFAULT_CONFIG, SpikingNavConfig
from spikingnav.nn.lif import SurrogateHeaviside


class SpikingPolicyNetwork(nn.Module):
    """Recurrent LIF policy core with linear actor / critic heads."""

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        cfg: Optional[SpikingNavConfig] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or DEFAULT_CONFIG
        hidden = self.cfg.hidden_size
        self.hidden_size = hidden
        self.w_r = nn.Linear(input_dim, hidden)
        self.w_h = nn.Linear(hidden, hidden, bias=False)
        self.actor = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.w_r.weight, gain=1.0)
        # leak * I + W_h must stay contractive across a 128-step rollout.
        nn.init.orthogonal_(self.w_h.weight, gain=0.05)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.w_r.bias)
        nn.init.zeros_(self.actor.bias)
        nn.init.zeros_(self.critic.bias)

    def step(
        self, r_t: torch.Tensor, u_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One environment step of Eqs. 7-11.

        Returns (logits, value, u_t).
        """
        c_sens = self.w_r(r_t)
        # Opt-in ablation; absent on historical/default checkpoints.
        if hasattr(self, "sensory_norm"):
            c_sens = self.sensory_scale * self.sensory_norm(c_sens)
        c_mem = self.w_h(u_prev)
        v_t = self.cfg.leak * u_prev + c_sens + c_mem
        s_t = SurrogateHeaviside.apply(
            v_t - self.cfg.threshold, self.cfg.surrogate_alpha
        )
        u_t = v_t * (1.0 - s_t)
        # Subthreshold membrane is otherwise unbounded below, and a 128-step
        # unroll overflows float32 when the recurrent gain exceeds 1.
        limit = 10.0 * self.cfg.threshold
        u_t = u_t.clamp(-limit, limit)
        logits = self.actor(u_t)
        value = self.critic(u_t)
        return logits, value, u_t

    def forward(
        self,
        r: torch.Tensor,
        u_prev: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unroll over environment steps.

        r: [T, N, D] or [N, D]
        u_prev: [N, H]
        masks: [T, N, 1] with 0 at episode starts
        """
        if r.dim() == 2:
            r = r.unsqueeze(0)
        t_steps, batch, _ = r.shape
        if masks is None:
            masks = r.new_ones(t_steps, batch, 1)
        elif masks.dim() == 2:
            masks = masks.unsqueeze(-1)

        logits = []
        values = []
        u = u_prev
        for t in range(t_steps):
            # Multiply would keep NaN alive (NaN * 0 = NaN) across episode resets.
            u = torch.where(masks[t] > 0, u, torch.zeros_like(u))
            logit, value, u = self.step(r[t], u)
            logits.append(logit)
            values.append(value)
        return torch.stack(logits, 0), torch.stack(values, 0), u
