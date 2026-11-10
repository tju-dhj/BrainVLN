"""Same RGB-D history, two instructions. Spatial should move less than Goal."""

from __future__ import annotations

import torch

from spikingnav.brain_vln.model import BrainVLNModel


def population_mean(membrane: torch.Tensor, bounds) -> torch.Tensor:
    start, stop = bounds
    return membrane[:, start:stop].mean(dim=-1)


def counterfactual_probe(
    model: BrainVLNModel,
    rgb: torch.Tensor,
    depth: torch.Tensor,
    depth_valid: torch.Tensor,
    tokens_a: torch.Tensor,
    mask_a: torch.Tensor,
    tokens_b: torch.Tensor,
    mask_b: torch.Tensor,
) -> dict:
    """Run two instructions on one shared observation sequence."""
    if model.config.variant not in ("B4", "B5"):
        raise ValueError("the separation probe is defined for B4 and B5")
    valid = torch.ones(rgb.shape[:2], dtype=torch.bool, device=rgb.device)
    reset = torch.zeros_like(valid)
    reset[0] = True
    detach = torch.zeros_like(valid)
    _, state_a, _ = model(rgb, depth, depth_valid, tokens_a, mask_a, valid, reset, detach)
    _, state_b, _ = model(rgb, depth, depth_valid, tokens_b, mask_b, valid, reset, detach)
    bounds = model.bounds
    spatial_a = population_mean(state_a.membrane, bounds["spatial"])
    spatial_b = population_mean(state_b.membrane, bounds["spatial"])
    goal_a = population_mean(state_a.membrane, bounds["goal"])
    goal_b = population_mean(state_b.membrane, bounds["goal"])
    return {
        "spatial_abs_diff": float((spatial_a - spatial_b).abs().mean()),
        "goal_abs_diff": float((goal_a - goal_b).abs().mean()),
        "spatial_equal": bool(torch.allclose(spatial_a, spatial_b, atol=1e-5, rtol=1e-4)),
        "goal_changed": bool(not torch.allclose(goal_a, goal_b, atol=1e-5, rtol=1e-4)),
    }
