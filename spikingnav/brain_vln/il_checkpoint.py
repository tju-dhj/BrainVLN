"""Choose the imitation checkpoint that MLANet would keep for PPO.

MLANet evaluates during imitation and selects the val-unseen checkpoint by
SPL. This helper reads finished ``summary.json`` files written by
``scripts/brain_vln/eval_val_unseen.py``. A partial episode log is not a
result: that script writes the summary only after the full split is scored.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


def select_best_il_checkpoint(eval_root: Path) -> tuple:
    """Return ``(checkpoint, summary)`` with the highest finished val-unseen SPL."""
    root = Path(eval_root)
    if not root.is_dir():
        raise FileNotFoundError(f"val-unseen eval directory does not exist: {root}")
    best = None
    best_key = None
    for summary_path in sorted(root.glob("update_*/summary.json")):
        row = json.loads(summary_path.read_text(encoding="utf-8"))
        spl = row.get("spl")
        checkpoint = row.get("checkpoint")
        if not isinstance(spl, (int, float)) or not math.isfinite(float(spl)):
            continue
        if not isinstance(checkpoint, str) or not Path(checkpoint).is_file():
            continue
        success = row.get("sr")
        success_rate = float(success) if isinstance(success, (int, float)) and math.isfinite(float(success)) else -1.0
        update = int(row.get("update", -1))
        key = (float(spl), success_rate, update)
        if best_key is None or key > best_key:
            best_key = key
            best = (Path(checkpoint), row)
    if best is None:
        raise FileNotFoundError(
            f"no finished val-unseen summary with a checkpoint under {root}. "
            "A summary is written only after the full split is scored. "
            "Pass --init to use a specific imitation checkpoint."
        )
    return best


def checkpoint_due(update: int, updates: int, checkpoints: int) -> bool:
    """True on the last update of each equal portion, including the final update."""
    if checkpoints < 1 or updates < 1:
        raise ValueError("updates and checkpoints must be positive")
    if not 0 <= update < updates:
        raise ValueError("update must lie in [0, updates)")
    previous = int(update / updates * checkpoints)
    current = int((update + 1) / updates * checkpoints)
    return current > previous


def shard_episodes(episodes: list, workers: int) -> list:
    """Give each environment a non-empty share, keeping whole scenes together."""
    if workers < 1:
        raise ValueError("workers must be positive")
    if len(episodes) < workers:
        raise ValueError("need at least one episode per environment")
    groups = []
    for episode in episodes:
        scene = Path(episode).parent.name
        if not groups or groups[-1][0] != scene:
            groups.append((scene, [episode]))
        else:
            groups[-1][1].append(episode)
    shards = [[] for _ in range(workers)]
    for index, (_scene, paths) in enumerate(groups):
        shards[index % workers].extend(paths)
    if any(not shard for shard in shards):
        shards = [[] for _ in range(workers)]
        for index, episode in enumerate(episodes):
            shards[index % workers].append(episode)
    return shards
