#!/usr/bin/env python3
"""Short model-level smoke test (no simulator required)."""

from __future__ import annotations

import argparse

import torch

from spikingnav.models.ann_nav import ANNNavModel
from spikingnav.models.actor_critic import SpikingNavModel
from spikingnav.utils.flops import count_parameters, estimate_step_flops, format_param_report


def _run(model: torch.nn.Module, rgb: torch.Tensor, goal: torch.Tensor) -> None:
    model.train()
    logits, values, memory = model(rgb, goal)
    loss = logits.mean() + values.mean()
    loss.backward()
    print(
        f"  logits={tuple(logits.shape)} values={tuple(values.shape)} "
        f"memory={tuple(memory.shape)} loss={float(loss.detach()):.4f}"
    )
    print(" ", format_param_report(model))
    flops = estimate_step_flops(
        model,
        time_steps=getattr(getattr(model, "cfg", None), "time_steps", 1),
        snn_backbone=isinstance(model, SpikingNavModel),
    )
    print(f"  estimated FLOPs/step = {flops['flops_g']:.3f}G")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    rgb = torch.randn(2, 2, 224, 224, 3, device=device)
    goal = torch.randint(0, 12, (2, 2), device=device)

    print("SpikingNav")
    snn = SpikingNavModel(action_dim=6, num_categories=12, pretrained_ann=False).to(device)
    _run(snn, rgb, goal)

    print("ANNNav")
    ann = ANNNavModel(
        action_dim=6, num_categories=12, freeze_backbone=False, pretrained=False
    ).to(device)
    _run(ann, rgb, goal)
    print("smoke test passed")


if __name__ == "__main__":
    main()
