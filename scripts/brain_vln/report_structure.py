#!/usr/bin/env python3
"""Structural ablation report. Navigation scores stay NOT_RUN until a real closed loop."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from spikingnav.brain_vln.config import BrainVLNConfig, PopulationSizes, tiny_config
from spikingnav.brain_vln.flywire import flywire_status
from spikingnav.brain_vln.lesion import population_mask, random_mask
from spikingnav.brain_vln.model import VARIANTS, BrainVLNModel
from spikingnav.brain_vln.synapses import build_graph


def _forward_stats(model, device) -> dict:
    model = model.to(device).eval()
    batch = 1
    rgb = torch.rand(1, batch, 3, 224, 224, device=device)
    depth = torch.rand(1, batch, 1, 224, 224, device=device)
    depth_valid = torch.ones(1, batch, 1, 224, 224, dtype=torch.bool, device=device)
    tokens = torch.randint(1, 32, (batch, 8), device=device)
    mask = torch.ones(batch, 8, dtype=torch.bool, device=device)
    valid = torch.ones(1, batch, dtype=torch.bool, device=device)
    reset = valid.clone()
    detach = torch.zeros_like(valid)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        _logits, state, aux = model(rgb, depth, depth_valid, tokens, mask, valid, reset, detach)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    spikes = state.spike
    silent = float((spikes.sum(dim=0) == 0).float().mean()) if spikes.numel() else 0.0
    return {
        "latency_s": elapsed,
        "peak_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else None
        ),
        "firing_rate": float(aux["firing_rate"].mean()),
        "silent_ratio": silent,
        "finite": bool(torch.isfinite(state.membrane).all()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error(f"{output} already exists")
    output.mkdir(parents=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    rows = []
    for variant in VARIANTS:
        model = BrainVLNModel(tiny_config(variant=variant))
        params = sum(parameter.numel() for parameter in model.parameters())
        row = {
            "variant": variant,
            "scale": "tiny",
            "parameters": params,
            "edges": None if model.graph is None else model.graph.n_edges,
            "synapse_statistics": None if model.graph is None else model.graph.statistics(),
            **_forward_stats(model, torch.device("cpu")),
            "sr": "NOT_RUN",
            "spl": "NOT_RUN",
            "ne": "NOT_RUN",
            "ndtw": "NOT_RUN",
        }
        rows.append(row)
        del model
    degree_rows = []
    for degree in (24, 48, 96):
        config = BrainVLNConfig(
            populations=PopulationSizes(spatial=64, goal=32, episodic=48, action=32),
            out_degree=degree,
            seed=0,
        )
        graph = build_graph(config, "random_sparse")
        degree_rows.append({"requested_out_degree": degree, **graph.statistics()})
    full = BrainVLNConfig(variant="B2", seed=0)
    full_model = BrainVLNModel(full)
    full_stats = {
        "variant": "B2",
        "scale": "4096",
        "neurons": full.populations.total,
        "parameters": sum(parameter.numel() for parameter in full_model.parameters()),
        "synapse_statistics": full_model.graph.statistics(),
        **_forward_stats(full_model, device),
        "sr": "NOT_RUN",
        "spl": "NOT_RUN",
        "ne": "NOT_RUN",
        "ndtw": "NOT_RUN",
    }
    goal = population_mask(full, "goal")
    report = {
        "navigation_metrics": "NOT_RUN",
        "reason": "no full closed-loop evaluation has been run for these checkpoints",
        "variants_tiny_cpu": rows,
        "out_degree_comparison": degree_rows,
        "default_b2": full_stats,
        "lesion_goal_count": int(goal.sum()),
        "lesion_random_count": int(random_mask(full, int(goal.sum()), seed=0).sum()),
        "flywire": flywire_status(Path("storage/brain_vln/flywire")),
        "stdp": "implemented separately from the imitation forward; NOT_RUN as a training experiment",
        "learnable_delay": "separate module; main-graph delays stay fixed; NOT_RUN as a training experiment",
        "dagger": "NOT_RUN",
    }
    (output / "ablation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"default_edges": full_stats["synapse_statistics"]["edges"], "finite": full_stats["finite"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
