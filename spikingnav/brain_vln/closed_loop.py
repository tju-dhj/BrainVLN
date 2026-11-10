"""Closed-loop actor. Metrics may use the goal; the actor cannot."""

from __future__ import annotations

from pathlib import Path

import torch

from spikingnav.brain_vln.model import BrainVLNModel
from spikingnav.vlnce.metrics import ndtw, path_length, spl


class ClosedLoopActor:
    """Choose an action from RGB-D and the instruction only."""

    def __init__(self, model: BrainVLNModel) -> None:
        self.model = model
        self.state = None
        self.prev_action = torch.tensor([4], dtype=torch.long)

    def reset(self) -> None:
        self.state = None
        self.prev_action = torch.tensor([4], dtype=torch.long)

    def act(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        depth_valid: torch.Tensor,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        instruction_text: str = None,
    ) -> int:
        if rgb.dim() == 3:
            rgb = rgb.unsqueeze(0)
            depth = depth.unsqueeze(0)
            depth_valid = depth_valid.unsqueeze(0)
        rgb = rgb.unsqueeze(0)
        depth = depth.unsqueeze(0)
        depth_valid = depth_valid.unsqueeze(0)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
            token_mask = token_mask.unsqueeze(0)
        device = next(self.model.parameters()).device
        valid = torch.ones(1, 1, dtype=torch.bool, device=device)
        reset = torch.ones(1, 1, dtype=torch.bool, device=device) if self.state is None else torch.zeros_like(valid)
        detach = torch.zeros_like(valid)
        logits, self.state, _aux = self.model(
            rgb.to(device),
            depth.to(device),
            depth_valid.to(device),
            tokens.to(device),
            token_mask.to(device),
            valid,
            reset,
            detach,
            prev_action=self.prev_action.to(device),
            state=self.state,
            instruction_text=None if instruction_text is None else [instruction_text],
        )
        choice = int(logits[-1, 0].argmax().item())
        self.prev_action = torch.tensor([choice], dtype=torch.long)
        return choice


def score_trajectory(
    positions,
    reference,
    stopped: bool,
    final_distance: float,
    start_geodesic: float,
    success_distance: float = 3.0,
) -> dict:
    success = bool(stopped and final_distance <= success_distance)
    return {
        "success": success,
        "spl": spl(success, start_geodesic, path_length(positions)),
        "navigation_error": float(final_distance),
        "ndtw": ndtw(positions, reference, success_distance),
        "path_length": path_length(positions),
        "stopped": bool(stopped),
        "steps": len(positions),
    }


def prepare_dagger(destination: Path, closed_loop_stable: bool) -> Path:
    """DAgger writes only under storage/brain_vln/dagger, and only after a stable loop."""
    root = Path("storage/brain_vln/dagger").resolve()
    destination = Path(destination).resolve()
    if destination != root and root not in destination.parents:
        raise ValueError(f"DAgger output must stay under {root}")
    if not closed_loop_stable:
        raise RuntimeError("DAgger waits until closed-loop imitation is stable")
    destination.mkdir(parents=True, exist_ok=True)
    return destination
