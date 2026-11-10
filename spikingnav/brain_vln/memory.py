"""Event signals that scale episodic writes without erasing history."""

from __future__ import annotations

import torch
import torch.nn as nn


class EventMemory(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        mode: str = "no-gate",
        threshold: float = 0.5,
        period: int = 4,
        max_staleness: int = 8,
        skip_inactive: bool = True,
    ) -> None:
        super().__init__()
        if mode not in ("no-gate", "periodic", "surprise", "task-aware"):
            raise ValueError(f"unknown event mode {mode}")
        self.mode = mode
        self.threshold = float(threshold)
        self.period = int(period)
        self.max_staleness = int(max_staleness)
        self.skip_inactive = skip_inactive
        self.predictor = nn.Linear(feature_dim, feature_dim)
        self.executed_writes = 0

    def forward(
        self,
        spatial_latent: torch.Tensor,
        previous_spatial: torch.Tensor,
        goal_latent: torch.Tensor,
        episodic_latent: torch.Tensor,
        observation: torch.Tensor,
        staleness: torch.Tensor,
        time: torch.Tensor,
        fsa_state: torch.Tensor | None,
        previous_fsa: torch.Tensor | None,
    ) -> dict:
        del episodic_latent, observation
        prediction = self.predictor(previous_spatial)
        surprise = (prediction - spatial_latent.detach()).pow(2).mean(dim=-1)
        task = _task_switch(goal_latent, fsa_state, previous_fsa)
        gate = self._gate(surprise, task, staleness, time)
        forced = staleness >= self.max_staleness
        gate = torch.where(forced, torch.ones_like(gate), gate)
        write = self._masked_write(spatial_latent, gate)
        return {
            "gate": gate,
            "surprise": surprise,
            "prediction": prediction,
            "task_switch": task,
            "write": write,
            "staleness": torch.where(gate > 0, torch.zeros_like(staleness), staleness + 1),
        }

    def _gate(
        self,
        surprise: torch.Tensor,
        task: torch.Tensor,
        staleness: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        del staleness
        if self.mode == "no-gate":
            return torch.ones_like(surprise)
        if self.mode == "periodic":
            return (torch.remainder(time, self.period) == 0).to(surprise.dtype)
        if self.mode == "surprise":
            return (surprise > self.threshold).to(surprise.dtype)
        return ((surprise > self.threshold) | (task > 0)).to(surprise.dtype)

    def _masked_write(self, value: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        open_rows = gate > 0
        if self.skip_inactive and not bool(open_rows.any()):
            return torch.zeros_like(value)
        self.executed_writes += int(open_rows.sum())
        return value * gate.unsqueeze(-1)


def _task_switch(goal_latent: torch.Tensor, fsa_state, previous_fsa) -> torch.Tensor:
    """Task switches come from an explicit FSA state.

    Collected episodes have no FSA labels, so the switch stays zero unless the
    caller supplies one. ``goal_latent`` is part of the legal input set and is
    not turned into a fabricated boundary.
    """
    if fsa_state is None or previous_fsa is None:
        return goal_latent.new_zeros(goal_latent.shape[0])
    return (fsa_state != previous_fsa).to(dtype=goal_latent.dtype)
