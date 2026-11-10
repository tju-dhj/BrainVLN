#!/usr/bin/env python3
"""Closed-loop R2R-CE evaluation of a teacher-forced SpikingNav checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from spikingnav.vlnce.constants import DEFAULT_DATASET, DEFAULT_SCENES, MAX_EPISODE_STEPS
from spikingnav.vlnce.episodes import load_split
from spikingnav.vlnce.provenance import write_provenance
from spikingnav.vlnce.sim import evaluate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--scenes", default=DEFAULT_SCENES)
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0, help="Habitat-Sim GPU. Use -1 when this machine has no CUDA EGL device.")
    parser.add_argument("--max-steps", type=int, default=MAX_EPISODE_STEPS)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error(f"{output} already exists")
    demos = load_split(args.dataset, args.split, args.limit)
    summary = evaluate(
        demos,
        args.scenes,
        args.checkpoint,
        str(output),
        gpu_device_id=args.gpu,
        max_steps=args.max_steps,
    )
    write_provenance(str(output), sys.argv)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
