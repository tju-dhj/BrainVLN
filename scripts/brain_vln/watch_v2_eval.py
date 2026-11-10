#!/usr/bin/env python3
"""Evaluate BrainVLN_v2 on val_unseen every 5 epochs from the 20th epoch.

Epochs in the checkpoint name are 0-based, so the 20th epoch is
``ckpt_epoch_19.pt`` at optimizer step 2000. The next runs are steps
2500, 3000, and so on. Training is not touched. One closed-loop eval
runs at a time. A finished split writes SR, SPL, NE and nDTW into the eval TensorBoard
directory. It does not write into the training event directory, because
a newer event file there makes TensorBoard ignore later training steps.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def _checkpoint_step(path: Path) -> int:
    """Epoch n is saved at optimizer step ``(n + 1) * 100``.

    The step is taken from the filename so the watcher does not read the
    662MB checkpoint into host memory while training is running.
    """
    epoch = int(path.stem.split("_")[-1])
    return (epoch + 1) * 100


def _pending(checkpoint_dir: Path, output: Path, start: int, every: int) -> list:
    found = []
    for path in sorted(checkpoint_dir.glob("ckpt_epoch_*.pt")):
        step = _checkpoint_step(path)
        if step < start or step % every != 0:
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
    root = Path("storage/brain_vln/experiments")
    checkpoint_dir = root / "BrainVLN_v2"
    output = root / "BrainVLN_v2_val_unseen"
    output.mkdir(parents=True, exist_ok=True)
    writers = [
        SummaryWriter(log_dir=str(output / "tb"), flush_secs=2),
    ]
    python = Path(sys.executable)
    seen = set()
    while True:
        for summary in sorted(output.glob("update_*/summary.json")):
            if summary in seen:
                continue
            for writer in writers:
                _write_tb(writer, summary)
            seen.add(summary)
            print(json.dumps({"logged": str(summary)}), flush=True)
        pending = _pending(checkpoint_dir, output, start=2000, every=500)
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
            "6",
        ]
        print(json.dumps({"start": step, "checkpoint": checkpoint.name}), flush=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n# start update={step}\n")
            handle.flush()
            completed = subprocess.run(command, cwd=Path(".").resolve(), env=env, stdout=handle, stderr=subprocess.STDOUT)
        print(json.dumps({"finish": step, "code": completed.returncode}), flush=True)
        time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
