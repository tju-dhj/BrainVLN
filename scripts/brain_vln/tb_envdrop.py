#!/usr/bin/env python3
"""Mirror EnvDrop collection and imitation steps into one TensorBoard directory.

Collection rows use ``kind=collect``. Training rows use the same fields as
the other BrainVLN runs: step, loss, accuracy, supervised_steps.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def _train_points(rows: dict) -> list:
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
                weighted_loss / max(supervised, 1),
                weighted_accuracy / max(supervised, 1),
            )
        )
    return points


def _ingest(path: Path, offset: int, rows: dict, kind: str) -> tuple:
    if not path.exists():
        return offset, False
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read()
    end = data.rfind(b"\n") + 1
    if not end:
        return offset, False
    changed = False
    for line in data[:end].splitlines():
        if not line:
            continue
        row = json.loads(line)
        if kind == "collect":
            if row.get("kind") != "collect" or "step" not in row:
                continue
            key = f"{row['stage']}:{int(row['step'])}"
        elif "loss" not in row or "step" not in row:
            continue
        else:
            key = int(row["step"])
        rows[key] = row
        changed = True
    return offset + end, changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-log", required=True)
    parser.add_argument("--collect-log", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output), flush_secs=2)
    train_rows = {}
    collect_rows = {}
    train_offset = 0
    collect_offset = 0
    written_train = set()
    written_collect = {}
    while True:
        train_offset, train_changed = _ingest(Path(args.train_log), train_offset, train_rows, "train")
        collect_offset, collect_changed = _ingest(Path(args.collect_log), collect_offset, collect_rows, "collect")
        if train_changed:
            for point in _train_points(train_rows):
                if point[0] in written_train:
                    continue
                step, loss, accuracy, loss_running, accuracy_running = point
                writer.add_scalar("train/loss", loss, step)
                writer.add_scalar("train/accuracy", accuracy, step)
                writer.add_scalar("train/loss_running", loss_running, step)
                writer.add_scalar("train/accuracy_running", accuracy_running, step)
                written_train.add(step)
        if collect_changed:
            for step, row in collect_rows.items():
                signature = (row["episodes"], row["episodes_per_second"])
                if written_collect.get(step) == signature:
                    continue
                tag = f"collect/{row['stage']}"
                scalar_step = int(row["step"])
                writer.add_scalar(tag, float(row["episodes"]), scalar_step)
                writer.add_scalar(f"{tag}_per_second", float(row["episodes_per_second"]), scalar_step)
                written_collect[step] = signature
        if train_changed or collect_changed:
            writer.flush()
            print(
                json.dumps({"train_steps": len(train_rows), "collect_points": len(collect_rows)}),
                flush=True,
            )
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
