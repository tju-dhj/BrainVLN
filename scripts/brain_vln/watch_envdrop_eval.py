#!/usr/bin/env python3
"""Evaluate BrainVLN_v2 EnvDrop on val_unseen every 5 epochs.

Checkpoints are 0-based, so epoch 5 is ``ckpt_envdrop_04.pt``. The runs are
epochs 5, 10, 15, 20, 25, 30, 35, 40 and 45. The watcher only launches
closed-loop eval. It does not signal the training process. Eval uses a
different GPU from the EnvDrop collector and trainer.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from spikingnav.brain_vln.mlanet_il import ENVDROP_EPOCHS


def _epoch(path: Path) -> int:
    prefix, epoch = path.stem.rsplit("_", 1)
    if prefix != "ckpt_envdrop":
        raise ValueError(path.name)
    return int(epoch)


def _ready(path: Path) -> bool:
    if not path.is_file():
        return False
    age = time.time() - path.stat().st_mtime
    return age >= 30 and path.stat().st_size > 100_000_000


def _pending(checkpoint_dir: Path, output: Path) -> list:
    found = []
    for path in sorted(checkpoint_dir.glob("ckpt_envdrop_*.pt")):
        try:
            epoch = _epoch(path)
        except ValueError:
            continue
        finished = epoch + 1
        if finished > ENVDROP_EPOCHS or finished % 5 != 0:
            continue
        if not _ready(path):
            continue
        summary = output / f"epoch_{finished:02d}" / "summary.json"
        if summary.is_file():
            continue
        found.append((finished, path))
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
    root = Path("storage/brain_vln/external_ssd1")
    checkpoint_dir = root / "BrainVLN_v2_envdrop_dagger_20261107"
    output = root / "BrainVLN_v2_envdrop_val_unseen"
    output.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output / "tb"), flush_secs=2)
    python = Path(sys.executable)
    gpu = 3
    seen = set()
    print(
        json.dumps(
            {
                "watching": str(checkpoint_dir),
                "output": str(output),
                "epochs": ENVDROP_EPOCHS,
                "every": 5,
                "gpu": gpu,
            }
        ),
        flush=True,
    )
    while True:
        for summary in sorted(output.glob("epoch_*/summary.json")):
            if summary in seen:
                continue
            _write_tb(writer, summary)
            seen.add(summary)
            print(json.dumps({"logged": str(summary)}), flush=True)
        pending = _pending(checkpoint_dir, output)
        if not pending:
            time.sleep(60)
            continue
        finished, checkpoint = pending[0]
        run_dir = output / f"epoch_{finished:02d}"
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
            str(finished),
            "--gpu",
            str(gpu),
        ]
        print(json.dumps({"start": finished, "checkpoint": checkpoint.name, "gpu": gpu}), flush=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        env["OMP_NUM_THREADS"] = "4"
        env["MKL_NUM_THREADS"] = "4"
        env["OPENBLAS_NUM_THREADS"] = "4"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n# start epoch={finished}\n")
            handle.flush()
            completed = subprocess.run(
                command,
                cwd=Path(".").resolve(),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        print(json.dumps({"finish": finished, "code": completed.returncode}), flush=True)
        time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
