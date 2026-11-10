#!/usr/bin/env python3
"""Tail a BrainVLN imitation jsonl log into TensorBoard events.

This process only reads the log. It does not touch the model or checkpoints.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def _points(rows: dict) -> list:
    """One point per step, in step order. A repeated step keeps the later row."""
    weighted_loss = 0.0
    weighted_accuracy = 0.0
    supervised = 0
    points = []
    for step in sorted(rows):
        row = rows[step]
        count = int(row["supervised_steps"])
        loss = float(row["loss"])
        accuracy = float(row["accuracy"])
        weighted_loss += loss * count
        weighted_accuracy += accuracy * count
        supervised += count
        points.append(
            (
                step,
                loss,
                accuracy,
                weighted_loss / supervised,
                weighted_accuracy / supervised,
            )
        )
    return points


def _write_events(output: Path, rows: dict):
    for path in output.glob("events.out.tfevents.*"):
        path.unlink()
    writer = SummaryWriter(log_dir=str(output), flush_secs=2)
    for step, loss, accuracy, loss_running, accuracy_running in _points(rows):
        writer.add_scalar("train/loss", loss, step)
        writer.add_scalar("train/accuracy", accuracy, step)
        writer.add_scalar("train/loss_running", loss_running, step)
        writer.add_scalar("train/accuracy_running", accuracy_running, step)
    writer.flush()
    return writer


def _add_point(writer, point) -> None:
    step, loss, accuracy, loss_running, accuracy_running = point
    writer.add_scalar("train/loss", loss, step)
    writer.add_scalar("train/accuracy", accuracy, step)
    writer.add_scalar("train/loss_running", loss_running, step)
    writer.add_scalar("train/accuracy_running", accuracy_running, step)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()
    log_path = Path(args.log)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = {}
    offset = 0
    writer = None
    max_step = -1
    while True:
        if not log_path.exists():
            time.sleep(args.interval)
            continue
        with log_path.open("rb") as handle:
            handle.seek(offset)
            data = handle.read()
        end = data.rfind(b"\n") + 1
        if not end:
            time.sleep(args.interval)
            continue
        offset += end
        fresh = []
        replaced = False
        for line in data[:end].splitlines():
            if not line:
                continue
            row = json.loads(line)
            if "step" not in row:
                continue
            step = int(row["step"])
            if step in rows or step < max_step:
                replaced = True
            elif step > max_step:
                fresh.append(step)
                max_step = step
            rows[step] = row
        if writer is not None and replaced:
            writer.close()
            writer = None
        if writer is None:
            writer = _write_events(output, rows)
        else:
            indexed = {point[0]: point for point in _points(rows)}
            for step in fresh:
                _add_point(writer, indexed[step])
            writer.flush()
        print(
            json.dumps({"mirrored_steps": len(rows), "max_step": max_step}),
            flush=True,
        )
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
