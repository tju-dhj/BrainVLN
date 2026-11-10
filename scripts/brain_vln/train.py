#!/usr/bin/env python3
"""Offline imitation for one BrainVLN variant. Does not launch a full corpus run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from spikingnav.brain_vln.config import BrainVLNConfig, PopulationSizes
from spikingnav.brain_vln.data.collate import collate_fn
from spikingnav.brain_vln.data.episode import load_episode_record
from spikingnav.brain_vln.model import BrainVLNModel
from spikingnav.brain_vln.train_loop import load_checkpoint, save_checkpoint, train_batch


def _config(args) -> BrainVLNConfig:
    if args.tiny:
        populations = PopulationSizes(spatial=8, goal=4, episodic=6, action=4)
        feature_dim = 16
        vocab_size = 8192
        out_degree = 2
    else:
        populations = PopulationSizes()
        feature_dim = 64
        vocab_size = 8192
        out_degree = 48
    return BrainVLNConfig(
        populations=populations,
        variant=args.variant,
        out_degree=out_degree,
        feature_dim=feature_dim,
        vocab_size=vocab_size,
        seed=args.seed,
        event_mode=args.event_mode,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", action="append", required=True, help="Collected episode directory. Repeat for a second episode.")
    parser.add_argument("--variant", default="B1", choices=["B0", "B1", "B2", "B3", "B4", "B5"])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--event-mode", default="surprise")
    parser.add_argument("--resume", default="")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and not args.resume:
        parser.error(f"{output} already exists")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    records = [load_episode_record(Path(path)) for path in args.episode]
    batch = collate_fn(records)
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    model = BrainVLNModel(_config(args)).to(device)
    optimizer = torch.optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.lr)
    start = 0
    if args.resume:
        payload = load_checkpoint(Path(args.resume), model, optimizer)
        start = int(payload["step"]) + 1
        model.to(device)
    history = []
    state = None
    for step in range(start, start + args.steps):
        state, parts = train_batch(model, batch, optimizer, grad_clip=args.grad_clip, state=None)
        parts["step"] = step
        history.append(parts)
        print(json.dumps(parts), flush=True)
    save_checkpoint(output / "ckpt_last.pt", model, optimizer, history[-1]["step"], history[-1])
    summary = {
        "variant": args.variant,
        "episodes": [str(Path(path)) for path in args.episode],
        "device": str(device),
        "steps": args.steps,
        "seed": args.seed,
        "tiny": bool(args.tiny),
        "weight_source": model.config.weight_source,
        "neuron_count": model.config.populations.total,
        "initial_loss": history[0]["loss"],
        "final_loss": history[-1]["loss"],
        "loss_decreased": history[-1]["loss"] < history[0]["loss"],
        "history": history,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("variant", "initial_loss", "final_loss", "loss_decreased")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
