#!/usr/bin/env python3
"""Train SpikingNav on HM3D ObjectNav with Habitat 0.3 (Habitat 3)."""

from __future__ import annotations

import argparse
import os

from spikingnav.habitat_objectnav.distributed import shutdown_distributed
from spikingnav.habitat_objectnav.il_trainer import HabitatILTrainer
from spikingnav.habitat_objectnav.trainer import HabitatSpikingTrainer

PPO_OUTPUT = "storage/habitat-objectnav-hm3d-spiking"
IL_OUTPUT = "storage/habitat-objectnav-hm3d-il-geodesic"
HUMAN_OUTPUT = "storage/habitat-objectnav-hm3d-il-human"
HUMAN_EPISODES = "datasets/objectnav_hm3d_hd/objectnav_hm3d_hd"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="ppo", choices=("ppo", "il", "il-human"))
    parser.add_argument("--split", default="train", choices=("train", "val", "val_mini"))
    parser.add_argument("--scenes", default="", help="Comma-separated content scene ids. Empty uses all.")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--total-steps", type=int, default=300_000_000)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--save-interval-updates", type=int, default=50)
    parser.add_argument(
        "--preemption-threshold",
        type=float,
        default=0.6,
        help="DD-PPO: fraction of workers that must finish a rollout before the rest may stop early. 0 disables it.",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--resume",
        default="",
        help="Checkpoint to continue from. Restores the network, the step count, and the optimizer when the file has one.",
    )
    parser.add_argument("--episode-root", default="")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--beta-start", type=float, default=1.0)
    parser.add_argument("--beta-end", type=float, default=0.0)
    parser.add_argument(
        "--beta-decay-steps",
        type=int,
        default=0,
        help="Steps to anneal expert mixture from beta-start to beta-end. 0 uses --total-steps.",
    )
    args = parser.parse_args()
    scenes = [part for part in args.scenes.split(",") if part]
    if args.output_dir:
        output_dir = args.output_dir
    elif args.algo == "il-human":
        output_dir = HUMAN_OUTPUT
    elif args.algo == "il":
        output_dir = IL_OUTPUT
    else:
        output_dir = PPO_OUTPUT
    trainer_cls = HabitatSpikingTrainer if args.algo == "ppo" else HabitatILTrainer
    il_kwargs = {}
    extra = {}
    if args.algo == "ppo":
        extra["preemption_threshold"] = args.preemption_threshold
    if args.episode_root:
        extra["episode_root"] = args.episode_root
    if args.algo == "il":
        il_kwargs = {
            "beta_start": args.beta_start,
            "beta_end": args.beta_end,
            "beta_decay_steps": args.beta_decay_steps,
            "expert": "geodesic",
        }
    elif args.algo == "il-human":
        il_kwargs = {
            "beta_start": 1.0,
            "beta_end": 1.0,
            "beta_decay_steps": 0,
            "expert": "human",
        }
        extra.setdefault("episode_root", os.path.abspath(HUMAN_EPISODES))
    trainer = trainer_cls(
        num_envs=args.num_envs,
        split=args.split,
        scenes=scenes or None,
        gpu=args.gpu,
        seed=args.seed,
        rollout_steps=args.rollout_steps,
        total_steps=args.total_steps,
        max_episode_steps=args.max_episode_steps,
        pretrained_ann=not args.no_pretrained and not args.resume,
        resume_path=args.resume,
        output_dir=output_dir,
        log_interval=args.log_interval,
        save_interval_updates=args.save_interval_updates,
        **extra,
        **il_kwargs,
    )
    try:
        trainer.train()
    except BaseException:
        shutdown_distributed()
        os._exit(1)
    # habitat-sim's static teardown aborts the process after a successful run.
    shutdown_distributed()
    os._exit(0)


if __name__ == "__main__":
    main()
