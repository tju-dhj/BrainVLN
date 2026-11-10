#!/usr/bin/env python3
"""Closed-loop BrainVLN evaluation on official R2R-CE val_unseen.

The actor sees RGB-D and the instruction text. The goal is used only to score
a finished trajectory. Success requires STOP within 3 m. A simulator or
instruction failure is recorded for that episode and is not turned into SR.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import sys
import traceback
from pathlib import Path

import torch

from spikingnav.brain_vln.closed_loop import ClosedLoopActor, score_trajectory
from spikingnav.brain_vln.config import BrainVLNConfig, PopulationSizes
from spikingnav.brain_vln.model import BrainVLNModel
from spikingnav.vlnce.constants import MAX_EPISODE_STEPS, SIM_ACTIONS, STOP
from spikingnav.vlnce.episodes import iter_split, mesh_path
from spikingnav.vlnce.sim import geodesic, make_simulator


def load_model(checkpoint: Path, device: torch.device) -> BrainVLNModel:
    payload = torch.load(checkpoint, map_location="cpu")
    raw = dict(payload["config"])
    populations = PopulationSizes(**raw.pop("populations"))
    raw.pop("neuron_count", None)
    model = BrainVLNModel(BrainVLNConfig(populations=populations, **raw))
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    model.frontend.set_frozen(True, True, True)
    return model


def _done_ids(path: Path) -> set:
    found = set()
    if not path.exists():
        return found
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("episode_id"):
            found.add(str(row["episode_id"]))
    return found


def _result_paths(output: Path) -> list:
    paths = []
    main = output / "episodes.jsonl"
    if main.exists():
        paths.append(main)
    paths.extend(sorted(output.glob("episodes.shard*.jsonl")))
    return paths


def _finished_ids(output: Path) -> set:
    found = set()
    for path in _result_paths(output):
        found |= _done_ids(path)
    return found


def _gather_rows(output: Path) -> list:
    rows = []
    seen = set()
    for path in _result_paths(output):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            episode_id = str(row.get("episode_id", ""))
            if not episode_id or episode_id in seen:
                continue
            seen.add(episode_id)
            rows.append(row)
    return rows


def _write_summary_if_complete(output: Path, update: int, checkpoint: str, expected: int) -> int:
    lock_path = output / ".merge.lock"
    with lock_path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        rows = _gather_rows(output)
        if len(rows) < expected:
            print(json.dumps({"partial": len(rows), "expected": expected}), flush=True)
            return 0
        summary = _summarize(rows)
        summary["update"] = update
        summary["checkpoint"] = checkpoint
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"summary": summary}), flush=True)
        return 0


def _observation(sim):
    from spikingnav.vlnce.sim import _depth, _position, _rgb

    rgb = torch.from_numpy(_rgb(sim)).permute(2, 0, 1).float().div(255.0)
    depth = torch.from_numpy(_depth(sim)).float()
    if depth.dim() == 3:
        depth = depth[..., 0]
    depth = depth.unsqueeze(0)
    valid = torch.isfinite(depth) & (depth > 0)
    depth = depth.masked_fill(~valid, 0)
    return rgb, depth, valid, _position(sim)


def _roll(sim, actor, demo, max_steps: int) -> dict:
    from spikingnav.vlnce.sim import _place

    _place(sim, demo["start_position"], demo["start_rotation"])
    tokens = torch.tensor(demo["tokens"], dtype=torch.long)
    mask = torch.ones_like(tokens, dtype=torch.bool)
    actor.reset()
    positions = []
    stopped = False
    for _step in range(max_steps):
        rgb, depth, valid, position = _observation(sim)
        positions.append(position)
        action = actor.act(
            rgb,
            depth,
            valid,
            tokens,
            mask,
            instruction_text=demo["instruction_text"],
        )
        if action == STOP:
            stopped = True
            break
        sim.step(SIM_ACTIONS[action])
    final = geodesic(sim, positions[-1], demo["goal"]) if positions else float("inf")
    scored = score_trajectory(
        positions,
        demo["reference_path"],
        stopped,
        final,
        float(demo["geodesic_distance"]),
    )
    scored["episode_id"] = demo["episode_id"]
    scored["scene_id"] = demo["scene_id"]
    scored["split"] = "val_unseen"
    return scored


def _summarize(rows: list) -> dict:
    scored = [row for row in rows if "error" not in row]
    count = len(scored)

    def mean(key):
        if not count:
            return None
        values = []
        for row in scored:
            value = row.get(key)
            if isinstance(value, bool):
                values.append(float(value))
            elif isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.append(float(value))
        if not values:
            return None
        return sum(values) / len(values)

    return {
        "split": "val_unseen",
        "protocol": "closed-loop R2R-CE",
        "episodes": len(rows),
        "scored": count,
        "errors": len(rows) - count,
        "sr": mean("success"),
        "spl": mean("spl"),
        "ne": mean("navigation_error"),
        "ndtw": mean("ndtw"),
        "success_distance_m": 3.0,
        "success_rule": "STOP called and geodesic distance <= 3 m",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--update", type=int, required=True)
    parser.add_argument("--dataset", default="storage/datasets/R2R_VLNCE_v1-3_preprocessed")
    parser.add_argument("--scenes", default="data/scene_datasets")
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=MAX_EPISODE_STEPS)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard < args.num_shards:
        raise SystemExit(f"invalid shard {args.shard}/{args.num_shards}")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    lock_name = ".eval.lock" if args.num_shards == 1 else f".eval.lock.{args.shard}"
    lock_handle = (output / lock_name).open("a", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(json.dumps({"busy": str(output), "shard": args.shard}), flush=True)
        return 0
    episodes_path = output / "episodes.jsonl" if args.num_shards == 1 else output / f"episodes.shard{args.shard:02d}.jsonl"
    done = _finished_ids(output)
    demos = []
    for index, demo in enumerate(iter_split(args.dataset, args.split)):
        if args.num_shards > 1 and index % args.num_shards != args.shard:
            continue
        if str(demo["episode_id"]) in done:
            continue
        demos.append(demo)
        if args.limit > 0 and len(done) + len(demos) >= args.limit:
            break
    if not demos:
        print(
            json.dumps({"update": args.update, "pending": 0, "already_done": len(done), "shard": args.shard}),
            flush=True,
        )
        if args.limit > 0:
            return 0
        expected = sum(1 for _ in iter_split(args.dataset, args.split))
        return _write_summary_if_complete(output, args.update, str(args.checkpoint), expected)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.checkpoint), device)
    actor = ClosedLoopActor(model)
    by_scene = {}
    for demo in demos:
        by_scene.setdefault(demo["scene_id"], []).append(demo)
    print(
        json.dumps(
            {
                "update": args.update,
                "pending": len(demos),
                "already_done": len(done),
                "shard": args.shard,
                "num_shards": args.num_shards,
                "device": str(device),
            }
        ),
        flush=True,
    )
    written = 0
    with episodes_path.open("a", encoding="utf-8") as handle:
        for scene_id, scene_demos in by_scene.items():
            mesh = mesh_path(args.scenes, scene_id)
            if not mesh.is_file():
                for demo in scene_demos:
                    row = {
                        "episode_id": demo["episode_id"],
                        "scene_id": scene_id,
                        "error": f"missing mesh {mesh}",
                    }
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    written += 1
                continue
            sim = make_simulator(str(mesh), args.gpu)
            try:
                for demo in scene_demos:
                    try:
                        with torch.inference_mode():
                            row = _roll(sim, actor, demo, args.max_steps)
                    except Exception as exc:
                        row = {
                            "episode_id": demo["episode_id"],
                            "scene_id": scene_id,
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                        }
                    row["update"] = args.update
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    written += 1
                    if written % 10 == 0 or "error" in row:
                        print(json.dumps({"finished": len(done) + written, "episode_id": row["episode_id"], "success": row.get("success"), "error": row.get("error"), "shard": args.shard}), flush=True)
            finally:
                sim.close()
    if args.limit > 0:
        print(json.dumps({"partial": len(done) + written, "shard": args.shard}), flush=True)
        return 0
    expected = sum(1 for _ in iter_split(args.dataset, args.split))
    return _write_summary_if_complete(output, args.update, str(args.checkpoint), expected)


if __name__ == "__main__":
    sys.exit(main())
