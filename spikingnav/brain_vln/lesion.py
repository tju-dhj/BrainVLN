"""Silence one population, or the same number of neurons chosen uniformly."""

from __future__ import annotations

import torch

from spikingnav.brain_vln.config import BrainVLNConfig


def population_mask(config: BrainVLNConfig, name: str) -> torch.Tensor:
    mask = torch.zeros(config.populations.total, dtype=torch.bool)
    start, stop = config.populations.bounds()[name]
    mask[start:stop] = True
    return mask


def random_mask(config: BrainVLNConfig, count: int, seed: int) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(seed)
    choice = torch.randperm(config.populations.total, generator=generator)[:count]
    mask = torch.zeros(config.populations.total, dtype=torch.bool)
    mask[choice] = True
    return mask


def apply_lesion(external: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return external.masked_fill(mask.to(device=external.device).view(1, -1), 0.0)
