#!/usr/bin/env python3
"""Teacher-forcing imitation of stored R2R-CE expert rollouts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from spikingnav.vlnce.constants import (
    DEFAULT_DATASET,
    IL_BATCH_SIZE,
    IL_EPOCHS,
    IL_LR,
    INFLECTION_COEF,
)
from spikingnav.vlnce.episodes import load_embeddings
from spikingnav.vlnce.il import train_offline
from spikingnav.vlnce.provenance import write_provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Directory of completed episode trees")
    parser.add_argument(
        "--embeddings",
        default=str(Path(DEFAULT_DATASET) / "embeddings.json.gz"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=IL_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=IL_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=IL_LR)
    parser.add_argument("--inflection", type=float, default=INFLECTION_COEF)
    parser.add_argument("--max-steps", type=int, default=0, help="Truncate each rollout. 0 keeps every expert step.")
    parser.add_argument("--limit", type=int, default=0, help="Use only the first N rollouts. 0 means all.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--device", default="")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error(f"{output} already exists")
    embeddings = load_embeddings(args.embeddings)
    result = train_offline(
        data=args.data,
        embeddings=embeddings,
        output_dir=str(output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        inflection=args.inflection,
        max_steps=args.max_steps,
        pretrained_ann=not args.no_pretrained,
        seed=args.seed,
        limit=args.limit,
        device=args.device,
    )
    write_provenance(str(result), sys.argv)
    print(f"checkpoint {result / 'ckpt_last.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
