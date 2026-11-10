#!/usr/bin/env python3
"""One-episode Habitat smoke for the BrainVLN actor.

The actor receives RGB-D and the instruction. Goal coordinates are used only
when scoring the finished trajectory. A failed simulator start is recorded
as an error, not as SR/SPL.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch

from spikingnav.brain_vln.closed_loop import ClosedLoopActor, score_trajectory
from spikingnav.brain_vln.config import tiny_config
from spikingnav.brain_vln.model import BrainVLNModel
from spikingnav.vlnce.constants import MAX_EPISODE_STEPS, SIM_ACTIONS, STOP
from spikingnav.vlnce.episodes import load_split, mesh_path, token_length
from spikingnav.vlnce.sim import geodesic, make_simulator


def _roll(sim, actor, demo, device, max_steps: int) -> dict:
    from spikingnav.vlnce.sim import _depth, _place, _position, _rgb

    _place(sim, demo["start_position"], demo["start_rotation"])
    tokens = torch.tensor(demo["tokens"], dtype=torch.long)
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    mask[: token_length(demo["tokens"])] = True
    actor.reset()
    positions = []
    stopped = False
    for _step in range(max_steps):
        positions.append(_position(sim))
        rgb = torch.from_numpy(_rgb(sim)).permute(2, 0, 1).float() / 255.0
        depth = torch.from_numpy(_depth(sim)).float()
        if depth.dim() == 3:
            depth = depth[..., 0]
        depth = depth.unsqueeze(0)
        depth_valid = torch.isfinite(depth) & (depth > 0)
        action = actor.act(rgb, depth, depth_valid, tokens, mask)
        if action == STOP:
            stopped = True
            break
        sim.step(SIM_ACTIONS[action])
    final = geodesic(sim, positions[-1], demo["goal"]) if positions else float("inf")
    scored = score_trajectory(
        positions,
        demo["locations"] or demo["reference_path"],
        stopped,
        final,
        float(demo["geodesic_distance"]),
    )
    scored["episode_id"] = demo["episode_id"]
    scored["scene_id"] = demo["scene_id"]
    scored["protocol"] = "smoke"
    return scored


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default="storage/datasets/R2R_VLNCE_v1-3_preprocessed")
    parser.add_argument("--scenes", default="data/scene_datasets")
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error(f"{output} already exists")
    output.mkdir(parents=True)
    summary = {
        "protocol": "closed-loop smoke",
        "max_steps": args.max_steps,
        "limit": args.limit,
        "sr": "NOT_RUN",
        "spl": "NOT_RUN",
        "ne": "NOT_RUN",
        "ndtw": "NOT_RUN",
    }
    try:
        demos = load_split(args.dataset, args.split, args.limit)
        if not demos:
            raise RuntimeError(f"no episodes in {args.split}")
        demo = demos[0]
        mesh = mesh_path(args.scenes, demo["scene_id"])
        if not mesh.is_file():
            raise FileNotFoundError(f"missing mesh {mesh}")
        device = torch.device("cpu")
        model = BrainVLNModel(tiny_config(variant="B0", vocab_size=8192)).to(device)
        actor = ClosedLoopActor(model)
        sim = make_simulator(str(mesh), args.gpu)
        try:
            row = _roll(sim, actor, demo, device, min(args.max_steps, MAX_EPISODE_STEPS))
        finally:
            sim.close()
        summary["smoke_trajectory"] = row
        summary["note"] = "two-step simulator wiring check; SR, SPL, NE and nDTW stay NOT_RUN"
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["traceback"] = traceback.format_exc()
        (output / "results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"error": summary["error"]}), flush=True)
        return 1
    (output / "results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"steps": summary.get("steps"), "stopped": summary.get("stopped")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
