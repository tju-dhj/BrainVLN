#!/usr/bin/env python3
"""Replay R2R-CE ground-truth actions into RGB-D-instruction-action trees."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from spikingnav.vlnce.constants import DEFAULT_DATASET, DEFAULT_SCENES
from spikingnav.vlnce.episodes import load_split, missing_meshes
from spikingnav.vlnce.provenance import write_provenance
from spikingnav.vlnce.sim import collect_split


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--scenes", default=DEFAULT_SCENES)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0, help="Replay at most this many episodes. 0 means all.")
    parser.add_argument("--gpu", type=int, default=0, help="Habitat-Sim GPU. Use -1 when this machine has no CUDA EGL device.")
    parser.add_argument("--shard", type=int, default=0, help="This worker's scene shard.")
    parser.add_argument("--num-shards", type=int, default=1, help="How many scene shards split the split.")
    parser.add_argument("--check-only", action="store_true", help="Only verify scene meshes. Does not start the simulator.")
    args = parser.parse_args()
    demos = load_split(args.dataset, args.split, args.limit)
    missing = missing_meshes(demos, args.scenes)
    print(
        f"episodes {len(demos)} missing_meshes {len(missing)} split {args.split}",
        flush=True,
    )
    if args.check_only:
        for episode_id, path in list(missing.items())[:20]:
            print(f"missing {episode_id} {path}")
        return 2 if missing else 0
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    provenance_dir = output / "logs" / f"shard{args.shard}"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    write_provenance(str(provenance_dir), sys.argv)
    summary = collect_split(
        demos,
        args.scenes,
        str(output),
        args.gpu,
        split=args.split,
        shard_index=args.shard,
        num_shards=args.num_shards,
    )
    print(summary, flush=True)
    return 2 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
