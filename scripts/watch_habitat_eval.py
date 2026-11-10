#!/usr/bin/env python3
"""Evaluate new Habitat checkpoints as training writes them.

Watches one checkpoint directory. Each finished ``ckpt_steps_*.pt`` is rolled
out on the validation split. Success rate and SPL are appended to
``eval_results.tsv``. After every evaluation only the ``--keep`` checkpoints
with the highest success rate are left on disk.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

from spikingnav.habitat_objectnav.ckpt_keeper import checkpoint_steps, rank_checkpoints

RESULT_FIELDS = ("time", "ckpt", "steps", "episodes", "sr", "spl")


def _result_path(ckpt_dir: str) -> str:
    return os.path.join(ckpt_dir, "eval_results.tsv")


def load_results(ckpt_dir: str) -> List[Dict]:
    path = _result_path(ckpt_dir)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def append_result(ckpt_dir: str, row: Dict) -> None:
    path = _result_path(ckpt_dir)
    new_file = not os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, delimiter="\t")
        if new_file:
            writer.writeheader()
        writer.writerow({key: row[key] for key in RESULT_FIELDS})
        handle.flush()


def pending_checkpoints(ckpt_dir: str, evaluated: set) -> List[str]:
    names = []
    for name in os.listdir(ckpt_dir):
        path = os.path.join(ckpt_dir, name)
        if not os.path.isfile(path) or name in evaluated:
            continue
        try:
            checkpoint_steps(path)
        except ValueError:
            continue
        names.append(path)
    return sorted(names, key=checkpoint_steps)


def file_is_stable(path: str, wait_s: float) -> bool:
    """True when the file does not change size across ``wait_s`` seconds."""
    if wait_s <= 0:
        return os.path.getsize(path) > 0
    first = os.path.getsize(path)
    if first <= 0:
        return False
    time.sleep(wait_s)
    return os.path.exists(path) and os.path.getsize(path) == first and first > 0


def prune_checkpoints(ckpt_dir: str, keep: int) -> List[str]:
    records = []
    for row in load_results(ckpt_dir):
        path = os.path.join(ckpt_dir, row["ckpt"])
        if not os.path.isfile(path):
            continue
        records.append(
            {
                "path": path,
                "ckpt": row["ckpt"],
                "sr": float(row["sr"]),
                "spl": float(row["spl"]),
                "steps": int(row["steps"]),
            }
        )
    kept, dropped = rank_checkpoints(records, keep)
    deleted = []
    for record in dropped:
        os.remove(record["path"])
        deleted.append(record["ckpt"])
        print(
            f"deleted {record['ckpt']} sr {record['sr']:.4f} spl {record['spl']:.4f}",
            flush=True,
        )
    if kept:
        summary = ", ".join(
            f"{record['ckpt']} sr {record['sr']:.4f}" for record in kept
        )
        print(f"keeping {summary}", flush=True)
    return deleted


def evaluate_checkpoint(
    path: str,
    num_envs: int,
    episodes: int,
    split: str,
    scenes: Optional[List[str]],
    gpu: int,
    seed: int,
    max_episode_steps: int,
) -> Dict[str, float]:
    import numpy as np
    import torch
    from habitat.core.vector_env import VectorEnv

    from spikingnav.config import DEFAULT_CONFIG
    from spikingnav.habitat_objectnav.make_env import (
        DEFAULT_EPISODE_ROOT,
        DEFAULT_SCENES_DIR,
        HM3D_CATEGORIES,
    )
    from spikingnav.habitat_objectnav.trainer import _prepare_rgb, _stack_obs, _worker_env
    from spikingnav.models.actor_critic import SpikingNavModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_fn_args = tuple(
        (
            split,
            scenes,
            gpu,
            seed,
            rank,
            max_episode_steps,
            DEFAULT_EPISODE_ROOT,
            DEFAULT_SCENES_DIR,
        )
        for rank in range(num_envs)
    )
    envs = VectorEnv(
        make_env_fn=_worker_env,
        env_fn_args=env_fn_args,
        auto_reset_done=True,
        multiprocessing_start_method="forkserver",
    )
    action_dim = int(envs.action_spaces[0].n)
    model = SpikingNavModel(
        action_dim=action_dim,
        num_categories=len(HM3D_CATEGORIES),
        cfg=DEFAULT_CONFIG,
        pretrained_ann=False,
    ).to(device)
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()
    observations = envs.reset()
    memory = model.initial_memory(envs.num_envs, device)
    masks = torch.ones(envs.num_envs, 1, device=device)
    success_sum = 0.0
    spl_sum = 0.0
    finished = 0
    while finished < episodes:
        rgb_np, goal_np = _stack_obs(observations)
        rgb = _prepare_rgb(torch.from_numpy(rgb_np).to(device), DEFAULT_CONFIG.image_size)
        goals = torch.from_numpy(goal_np).to(device)
        with torch.no_grad():
            logits, _, memory = model(rgb, goals, memory, masks.view(1, -1, 1))
            actions = torch.argmax(logits[-1], dim=-1)
        step_result = envs.step(actions.detach().cpu().numpy())
        observations = [item[0] for item in step_result]
        dones = np.asarray([item[2] for item in step_result], dtype=np.bool_)
        infos = [item[3] for item in step_result]
        masks = torch.from_numpy((~dones).astype(np.float32)).to(device).unsqueeze(-1)
        memory = memory * masks
        for info, done in zip(infos, dones):
            if not done:
                continue
            finished += 1
            success_sum += float(info.get("success", 0.0))
            spl_sum += float(info.get("spl", 0.0))
            if finished >= episodes:
                break
    metrics = {
        "episodes": finished,
        "sr": success_sum / finished if finished else 0.0,
        "spl": spl_sum / finished if finished else 0.0,
    }
    print("EVAL_JSON " + json.dumps(metrics), flush=True)
    os._exit(0)


def _metrics_from_output(text: str) -> Optional[Dict]:
    for line in reversed(text.splitlines()):
        if line.startswith("EVAL_JSON "):
            return json.loads(line[len("EVAL_JSON ") :])
    return None


def run_eval_subprocess(args: argparse.Namespace, path: str) -> Optional[Dict]:
    command = [
        sys.executable,
        os.path.abspath(__file__),
        "--eval-checkpoint",
        path,
        "--split",
        args.split,
        "--scenes",
        args.scenes,
        "--episodes",
        str(args.episodes),
        "--num-envs",
        str(args.num_envs),
        "--gpu",
        str(args.gpu),
        "--seed",
        str(args.seed),
        "--max-episode-steps",
        str(args.max_episode_steps),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    metrics = _metrics_from_output(completed.stdout)
    if metrics is None:
        print(f"eval failed for {os.path.basename(path)}", flush=True)
    return metrics


def watch(args: argparse.Namespace) -> None:
    ckpt_dir = os.path.abspath(args.ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(
        f"watching {ckpt_dir} split {args.split} episodes {args.episodes} keep {args.keep}",
        flush=True,
    )
    while True:
        evaluated = {row["ckpt"] for row in load_results(ckpt_dir)}
        progressed = False
        for path in pending_checkpoints(ckpt_dir, evaluated):
            if not os.path.exists(path) or not file_is_stable(path, args.stable_seconds):
                continue
            name = os.path.basename(path)
            print(f"eval {name}", flush=True)
            metrics = run_eval_subprocess(args, path)
            if metrics is None or not os.path.exists(path):
                continue
            row = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "ckpt": name,
                "steps": checkpoint_steps(path),
                "episodes": int(metrics["episodes"]),
                "sr": f"{float(metrics['sr']):.6f}",
                "spl": f"{float(metrics['spl']):.6f}",
            }
            append_result(ckpt_dir, row)
            print(
                f"recorded {name} episodes {row['episodes']} "
                f"sr {float(metrics['sr']):.4f} spl {float(metrics['spl']):.4f}",
                flush=True,
            )
            prune_checkpoints(ckpt_dir, args.keep)
            progressed = True
        evaluated_now = {row["ckpt"] for row in load_results(ckpt_dir)}
        if args.once and (progressed or not pending_checkpoints(ckpt_dir, evaluated_now)):
            return
        time.sleep(args.poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", default="")
    parser.add_argument("--split", default="val", choices=("train", "val", "val_mini"))
    parser.add_argument("--scenes", default="")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--keep", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument(
        "--stable-seconds",
        type=float,
        default=5.0,
        help="Wait this long and require the file size to stay unchanged before loading.",
    )
    parser.add_argument("--once", action="store_true", help="Eval the current queue, then exit.")
    parser.add_argument("--eval-checkpoint", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.eval_checkpoint:
        scenes = [part for part in args.scenes.split(",") if part] or None
        evaluate_checkpoint(
            args.eval_checkpoint,
            num_envs=args.num_envs,
            episodes=args.episodes,
            split=args.split,
            scenes=scenes,
            gpu=args.gpu,
            seed=args.seed,
            max_episode_steps=args.max_episode_steps,
        )
        return
    if not args.ckpt_dir:
        parser.error("the following arguments are required: --ckpt-dir")
    watch(args)


if __name__ == "__main__":
    main()
