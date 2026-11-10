#!/usr/bin/env python3
"""Evaluate each finished DAgger iteration on val_unseen.

There are 10 iterations and 4 epochs in each one. The watcher evaluates the
last epoch, ``ckpt_dagger_XX_03.pt``, once per iteration. Training is left
alone. One closed-loop eval runs at a time. SR, SPL, NE and nDTW go to a
separate TensorBoard directory so the training event file stays intact.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from spikingnav.brain_vln.mlanet_il import DAGGER_EPOCHS, DAGGER_ITERATIONS


def _iteration(path: Path) -> tuple:
    prefix, iteration, epoch = path.stem.rsplit("_", 2)
    if prefix != "ckpt_dagger":
        raise ValueError(path.name)
    return int(iteration), int(epoch)


def _ready(path: Path) -> bool:
    if not path.is_file():
        return False
    age = time.time() - path.stat().st_mtime
    return age >= 30 and path.stat().st_size > 100_000_000


def _pending(checkpoint_dir: Path, output: Path) -> list:
    last_epoch = DAGGER_EPOCHS - 1
    found = []
    for path in sorted(checkpoint_dir.glob("ckpt_dagger_*.pt")):
        try:
            iteration, epoch = _iteration(path)
        except ValueError:
            continue
        if epoch != last_epoch or not 0 <= iteration < DAGGER_ITERATIONS:
            continue
        if not _ready(path):
            continue
        summary = output / f"iter_{iteration:02d}" / "summary.json"
        if summary.is_file():
            continue
        found.append((iteration, path))
    found.sort(key=lambda item: item[0])
    return found


def _write_tb(writer: SummaryWriter, summary_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    step = int(summary["update"])
    for tag in ("sr", "spl", "ne", "ndtw"):
        value = summary.get(tag)
        if isinstance(value, (int, float)):
            writer.add_scalar(f"val_unseen/{tag}", float(value), step)
    writer.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        default="storage/brain_vln/external_ssd1/BrainVLN_v2_dagger_20261108",
    )
    parser.add_argument(
        "--output",
        default="storage/brain_vln/external_ssd1/BrainVLN_v2_dagger_val_unseen",
    )
    parser.add_argument("--gpu", type=int, default=6)
    args = parser.parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    output = Path(args.output)
    gpu = int(args.gpu)
    output.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output / "tb"), flush_secs=2)
    python = Path(sys.executable)
    seen = set()
    print(
        json.dumps(
            {
                "watching": str(checkpoint_dir),
                "output": str(output),
                "iterations": DAGGER_ITERATIONS,
                "epoch": DAGGER_EPOCHS - 1,
                "gpu": gpu,
            }
        ),
        flush=True,
    )
    while True:
        for summary in sorted(output.glob("iter_*/summary.json")):
            if summary in seen:
                continue
            _write_tb(writer, summary)
            seen.add(summary)
            print(json.dumps({"logged": str(summary)}), flush=True)
        pending = _pending(checkpoint_dir, output)
        if not pending:
            time.sleep(60)
            continue
        iteration, checkpoint = pending[0]
        run_dir = output / f"iter_{iteration:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "stdout.log"
        command = [
            str(python),
            "-u",
            "scripts/brain_vln/eval_val_unseen.py",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(run_dir),
            "--update",
            str(iteration),
            "--gpu",
            str(gpu),
        ]
        print(json.dumps({"start": iteration, "checkpoint": checkpoint.name}), flush=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        env["OMP_NUM_THREADS"] = "4"
        env["MKL_NUM_THREADS"] = "4"
        env["OPENBLAS_NUM_THREADS"] = "4"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n# start iteration={iteration}\n")
            handle.flush()
            completed = subprocess.run(
                command,
                cwd=Path(".").resolve(),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        print(json.dumps({"finish": iteration, "code": completed.returncode}), flush=True)
        time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
