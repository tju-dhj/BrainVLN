#!/usr/bin/env python3
"""Evaluate BrainVLN_v5 on val_unseen every 5 epochs.

Epochs in the checkpoint name are 0-based, so the 5th epoch is
``ckpt_epoch_4.pt`` at optimizer step 500. The runs are steps 500, 1000,
and so on through 10000. Training is not touched. One closed-loop eval
runs at a time. SR, SPL, NE and nDTW go to a separate TensorBoard directory
so the training event file stays intact.
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


def _checkpoint_step(path: Path) -> int:
    """Epoch n is saved at optimizer step ``(n + 1) * 100``."""
    epoch = int(path.stem.split("_")[-1])
    return (epoch + 1) * 100


def _ready(path: Path) -> bool:
    if not path.is_file():
        return False
    age = time.time() - path.stat().st_mtime
    return age >= 30 and path.stat().st_size > 100_000_000


def _pending(checkpoint_dir: Path, output: Path, start: int, every: int) -> list:
    found = []
    for path in sorted(checkpoint_dir.glob("ckpt_epoch_*.pt")):
        step = _checkpoint_step(path)
        if step < start or step % every != 0 or not _ready(path):
            continue
        summary = output / f"update_{step:06d}" / "summary.json"
        if summary.is_file():
            continue
        found.append((step, path))
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
        default="storage/brain_vln/external_ssd1/BrainVLN_v5",
    )
    parser.add_argument(
        "--output",
        default="storage/brain_vln/external_ssd1/BrainVLN_v5_val_unseen",
    )
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--start", type=int, default=500)
    parser.add_argument("--every", type=int, default=500)
    args = parser.parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output / "tb"), flush_secs=2)
    python = Path(sys.executable)
    seen = set()
    print(
        json.dumps(
            {
                "watching": str(checkpoint_dir),
                "output": str(output),
                "start": args.start,
                "every": args.every,
                "gpu": args.gpu,
            }
        ),
        flush=True,
    )
    while True:
        for summary in sorted(output.glob("update_*/summary.json")):
            if summary in seen:
                continue
            _write_tb(writer, summary)
            seen.add(summary)
            print(json.dumps({"logged": str(summary)}), flush=True)
        pending = _pending(checkpoint_dir, output, args.start, args.every)
        if not pending:
            time.sleep(60)
            continue
        step, checkpoint = pending[0]
        run_dir = output / f"update_{step:06d}"
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
            str(step),
            "--gpu",
            str(args.gpu),
        ]
        print(json.dumps({"start": step, "checkpoint": checkpoint.name, "gpu": args.gpu}), flush=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        env["OMP_NUM_THREADS"] = "4"
        env["MKL_NUM_THREADS"] = "4"
        env["OPENBLAS_NUM_THREADS"] = "4"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n# start update={step}\n")
            handle.flush()
            completed = subprocess.run(
                command,
                cwd=Path(".").resolve(),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        print(json.dumps({"finish": step, "code": completed.returncode}), flush=True)
        time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
