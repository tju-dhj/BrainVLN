#!/usr/bin/env python3
"""Train SpikingNav by behavior cloning on stored human demonstrations."""

from __future__ import annotations

import argparse

from spikingnav.habitat_objectnav.offline_il import train_offline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="storage/il_offline/hm3d_hd/train")
    parser.add_argument("--output-dir", default="storage/habitat-objectnav-hm3d-il-offline")
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    train_offline(
        root=args.data,
        output_dir=args.output_dir,
        steps=args.steps,
        batch_size=args.batch_size,
        window=args.window,
        pretrained_ann=not args.no_pretrained,
    )


if __name__ == "__main__":
    main()
