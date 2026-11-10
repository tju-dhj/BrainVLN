#!/usr/bin/env python3
"""EnvDrop teacher forcing, then DAgger, starting from a BrainVLN_v2 checkpoint.

The schedule matches MLANet's released configs:

- ``mla_aug.yaml``: 45 epochs, Adam 2.5e-4, inflection weight 3.2, one
  iteration, ``p = 1`` (teacher forcing) on ``joint_train_envdrop``.
- ``mla_aug_da_tune.yaml``: load those weights, then 10 iterations of 4
  epochs. Each iteration appends 5000 train-split rollouts. The expert
  action is mixed in with probability ``0.5 ** (iteration + 1)``.

Supervision is the same weighted cross-entropy as MLANet's
``BaseVLNCETrainer._update_agent``. Frozen MLANet maps are stored once per
episode so the 45 epochs do not re-render. The reservoir batch is wider than
MLANet's 5 trajectories; the epoch count, learning rate, loss, and DAgger
schedule are unchanged. See the provenance file for that batch-size note.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from spikingnav.brain_vln.config import BrainVLNConfig, PopulationSizes
from spikingnav.brain_vln.mlanet_il import (
    DAGGER_EPOCHS,
    DAGGER_ITERATIONS,
    DAGGER_P,
    DAGGER_UPDATE_SIZE,
    ENVDROP_EPOCHS,
    ENVDROP_LR,
    ENVDROP_P,
    ENVDROP_UPDATE_SIZE,
    INFLECTION_COEF,
    START_ACTION,
    dagger_beta,
    inflection_weights,
    partition_episodes,
    select_dagger_episodes,
    shard_dagger_demos,
    weighted_action_loss,
)
from spikingnav.brain_vln.model import BrainVLNModel
from spikingnav.brain_vln.train_loop import load_checkpoint, save_checkpoint
from spikingnav.vlnce.constants import (
    MAX_EPISODE_STEPS,
    MOVE_FORWARD,
    SIM_ACTIONS,
    STOP,
    TURN_LEFT,
    TURN_RIGHT,
)
from spikingnav.vlnce.episodes import load_json_gz, mesh_path
from spikingnav.vlnce.store import scene_name

_ACTION_NAME = {
    "stop": STOP,
    "move_forward": MOVE_FORWARD,
    "turn_left": TURN_LEFT,
    "turn_right": TURN_RIGHT,
}


def _git(args: list) -> str:
    try:
        result = subprocess.run(["git", *args], check=False, capture_output=True, text=True)
    except OSError as exc:
        return f"unavailable: {exc}"
    text = (result.stdout or result.stderr).strip()
    return text or f"exit {result.returncode}"


def _source_hash() -> str:
    digest = hashlib.sha256()
    roots = [
        Path("spikingnav/brain_vln"),
        Path("scripts/brain_vln/train_envdrop_dagger.py"),
    ]
    files = []
    for root in roots:
        if root.is_file():
            files.append(root)
        else:
            files.extend(path for path in root.rglob("*.py") if path.is_file())
    for path in sorted(files):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _fit_tokens(instruction_tokens, sub_rows):
    dped = Path("/home/vln/dhj/DPedPro-VLN")
    if str(dped) not in sys.path:
        sys.path.insert(0, str(dped))
    from vlnce.mlanet_instructions import fit_sub_rows, r2r_tokens

    return r2r_tokens(instruction_tokens, 2504), fit_sub_rows(sub_rows)


def load_demos(dataset_root: Path, split: str, need_actions: bool) -> list:
    sub = load_json_gz(dataset_root / split / f"{split}_sub.json.gz")
    ground_truth = load_json_gz(dataset_root / split / f"{split}_gt.json.gz") if need_actions else {}
    episodes = sub["episodes"] if isinstance(sub, dict) else sub
    demos = []
    missing_gt = 0
    for episode in episodes:
        key = str(episode["episode_id"])
        instruction = episode.get("instruction") or {}
        record = {
            "episode_id": key,
            "scene_id": str(episode["scene_id"]),
            "instruction_text": str(instruction.get("instruction_text", "")),
            "instruction_tokens": [int(token) for token in instruction.get("instruction_tokens") or []],
            "sub_instruction_tokens": episode.get("sub_instruction_tokens") or [],
            "start_position": [float(value) for value in episode["start_position"]],
            "start_rotation": [float(value) for value in episode["start_rotation"]],
            "goal": [float(value) for value in episode["goals"][0]["position"]],
        }
        if need_actions:
            if key not in ground_truth:
                missing_gt += 1
                continue
            record["actions"] = [int(action) for action in ground_truth[key]["actions"]]
            if not record["actions"]:
                missing_gt += 1
                continue
        demos.append(record)
    if missing_gt:
        print(f"{split} skipped {missing_gt} episodes without ground-truth actions", flush=True)
    return demos


def _cached_demos(path: Path, dataset_root: Path, split: str, need_actions: bool) -> list:
    if path.is_file():
        with path.open("rb") as handle:
            return pickle.load(handle)
    demos = load_demos(dataset_root, split, need_actions)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(demos, handle, protocol=4)
    return demos


def _observation(sim):
    from spikingnav.vlnce.sim import _depth, _rgb

    rgb = torch.from_numpy(_rgb(sim)).permute(2, 0, 1).float().div(255.0)
    depth = torch.from_numpy(_depth(sim)).float()
    if depth.dim() == 3:
        depth = depth[..., 0]
    depth = depth.unsqueeze(0)
    valid = torch.isfinite(depth) & (depth > 0)
    depth = depth.masked_fill(~valid, 0)
    return rgb, depth, valid


def _expert_action(sim, follower, goal) -> int:
    import habitat_sim

    agent = sim.get_agent(0)
    before = agent.get_state()
    position = np.array(before.position, dtype=np.float32, copy=True)
    rotation = before.rotation
    try:
        name = follower.next_action_along(np.asarray(goal, dtype=np.float32))
    except Exception:
        name = "stop"
    restored = habitat_sim.AgentState()
    restored.position = position
    restored.rotation = rotation
    agent.set_state(restored)
    if name is None:
        return STOP
    return _ACTION_NAME.get(str(name), STOP)


def _make_follower(sim):
    import habitat_sim

    return habitat_sim.GreedyGeodesicFollower(
        sim.pathfinder,
        sim.get_agent(0),
        goal_radius=0.5,
        stop_key="stop",
        forward_key="move_forward",
        left_key="turn_left",
        right_key="turn_right",
    )


def _language(model, demo, device: torch.device) -> torch.Tensor:
    tokens, rows = _fit_tokens(demo["instruction_tokens"], demo["sub_instruction_tokens"])
    instruction = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    sub = torch.tensor(rows, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        return model.frontend.pretrained.language_from_tokens(instruction, sub)[0].detach().cpu()


def _encode_frames(model, frames, depths, valids, device: torch.device, chunk: int = 16):
    rgb_maps = []
    depth_maps = []
    with torch.no_grad():
        for start in range(0, len(frames), chunk):
            rgb = torch.stack(frames[start : start + chunk]).to(device)
            depth = torch.stack(depths[start : start + chunk]).to(device)
            valid = torch.stack(valids[start : start + chunk]).to(device)
            rgb_map, depth_map = model.frontend.pretrained.encode_visual_maps(rgb, depth, valid)
            rgb_maps.append(rgb_map.detach().cpu())
            depth_maps.append(depth_map.detach().cpu())
    return torch.cat(rgb_maps, dim=0), torch.cat(depth_maps, dim=0)


def _save_feature(path: Path, rgb_map, depth_map, language, prev, oracle) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        rgb=rgb_map.numpy().astype(np.float16),
        depth=depth_map.numpy().astype(np.float16),
        language=language.numpy().astype(np.float16),
        prev=np.asarray(prev, dtype=np.int64),
        oracle=np.asarray(oracle, dtype=np.int64),
    )
    temporary.replace(path)


def _feature_path(root: Path, demo: dict) -> Path:
    return root / scene_name(demo["scene_id"]) / f"{demo['episode_id']}.npz"


def _append_index(index_path: Path, path: Path, demo: dict, steps: int) -> None:
    record = {
        "path": str(path),
        "steps": int(steps),
        "episode_id": demo["episode_id"],
        "scene_id": demo["scene_id"],
    }
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _reconcile_index(index_path: Path, feature_root: Path) -> list:
    rows = _read_index(index_path)
    known = {str(Path(row["path"]).resolve()) for row in rows}
    for path in sorted(feature_root.rglob("*.npz")):
        if path.name.endswith(".tmp.npz"):
            continue
        resolved = str(path.resolve())
        if resolved in known:
            continue
        with np.load(path) as data:
            steps = int(np.asarray(data["oracle"]).shape[0])
        record = {
            "path": str(path),
            "steps": steps,
            "episode_id": path.stem,
            "scene_id": path.parent.name,
        }
        rows.append(record)
        _append_index(index_path, path, record, steps)
    return [row for row in rows if Path(row["path"]).is_file()]


def _read_index(index_path: Path) -> list:
    if not index_path.is_file():
        return []
    rows = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _grouped(demos: list) -> list:
    grouped = {}
    for demo in demos:
        grouped.setdefault(demo["scene_id"], []).append(demo)
    return [grouped[key] for key in sorted(grouped)]


def _replay_teacher(sim, demo) -> tuple:
    from spikingnav.vlnce.sim import _place

    _place(sim, demo["start_position"], demo["start_rotation"])
    frames, depths, valids, actions = [], [], [], []
    for action in demo["actions"]:
        action = int(action)
        if action not in SIM_ACTIONS and action != STOP:
            raise ValueError(f"unknown action {action}")
        rgb, depth, valid = _observation(sim)
        frames.append(rgb)
        depths.append(depth)
        valids.append(valid)
        actions.append(action)
        if action == STOP or len(actions) >= MAX_EPISODE_STEPS:
            break
        sim.step(SIM_ACTIONS[action])
    prev = [START_ACTION] + actions[:-1]
    return frames, depths, valids, prev, actions


def _rollout_dagger(sim, model, demo, beta: float, rng: random.Random, device: torch.device):
    from spikingnav.vlnce.sim import _place

    _place(sim, demo["start_position"], demo["start_rotation"])
    follower = _make_follower(sim)
    language = _language(model, demo, device).to(device).unsqueeze(0)
    state = None
    prev_action = START_ACTION
    rgb_parts, depth_parts = [], []
    prevs, oracles, executed = [], [], []
    for _step in range(MAX_EPISODE_STEPS):
        rgb, depth, valid = _observation(sim)
        rgb_map, depth_map = _encode_frames(model, [rgb], [depth], [valid], device, chunk=1)
        rgb_parts.append(rgb_map)
        depth_parts.append(depth_map)
        reset = torch.ones(1, 1, dtype=torch.bool, device=device) if state is None else torch.zeros(
            1, 1, dtype=torch.bool, device=device
        )
        ones = torch.ones(1, 1, dtype=torch.bool, device=device)
        zeros = torch.zeros(1, 1, dtype=torch.bool, device=device)
        prev = torch.tensor([[prev_action]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, state, _aux = model.forward_from_maps(
                rgb_map.unsqueeze(0).to(device),
                depth_map.unsqueeze(0).to(device),
                language,
                prev,
                ones,
                reset,
                zeros,
                state=state,
            )
        student = int(torch.multinomial(torch.softmax(logits[-1, 0], dim=-1), 1).item())
        expert = _expert_action(sim, follower, demo["goal"])
        action = expert if rng.random() < beta else student
        prevs.append(prev_action)
        oracles.append(expert)
        executed.append(action)
        if action == STOP:
            break
        sim.step(SIM_ACTIONS[action])
        prev_action = action
    return torch.cat(rgb_parts), torch.cat(depth_parts), language[0].cpu(), prevs, oracles, executed


def _append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _log_collect(path: Path, stage: str, episodes: int, rate: float) -> None:
    _append_row(
        path,
        {
            "kind": "collect",
            "stage": stage,
            "step": int(episodes),
            "episodes": int(episodes),
            "episodes_per_second": float(rate),
        },
    )


def _next_train_step(path: Path) -> int:
    step = 0
    if not path.is_file():
        return 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "step" in row and row.get("kind") != "collect":
            step = max(step, int(row["step"]) + 1)
    return step


def collect_teacher(
    model,
    demos,
    feature_root: Path,
    index_path: Path,
    scenes: str,
    gpu: int,
    errors: Path,
    collect_log: Path,
) -> int:
    from spikingnav.vlnce.sim import make_simulator

    done = {row["episode_id"] + "@" + row["scene_id"] for row in _read_index(index_path)}
    pending = [demo for demo in demos if demo["episode_id"] + "@" + demo["scene_id"] not in done]
    print(f"envdrop collect pending {len(pending)} of {len(demos)}", flush=True)
    saved = 0
    started = time.time()
    _log_collect(collect_log, "envdrop", len(done), 0.0)
    device = next(model.parameters()).device
    for scene_demos in _grouped(pending):
        needed = [demo for demo in scene_demos if not _feature_path(feature_root, demo).is_file()]
        if not needed:
            continue
        mesh = mesh_path(scenes, needed[0]["scene_id"])
        if not mesh.is_file():
            _log_error(errors, needed[0], f"missing mesh {mesh}")
            continue
        sim = make_simulator(str(mesh), gpu)
        try:
            for demo in needed:
                destination = _feature_path(feature_root, demo)
                try:
                    frames, depths, valids, prev, oracle = _replay_teacher(sim, demo)
                    language = _language(model, demo, device)
                    rgb_map, depth_map = _encode_frames(model, frames, depths, valids, device)
                    _save_feature(destination, rgb_map, depth_map, language, prev, oracle)
                    _append_index(index_path, destination, demo, len(oracle))
                    saved += 1
                    if saved % 25 == 0:
                        rate = saved / max(time.time() - started, 1e-6)
                        print(f"envdrop collected {saved} {rate:.2f} eps/s", flush=True)
                        _log_collect(collect_log, "envdrop", len(done) + saved, rate)
                except Exception as exc:
                    _log_error(errors, demo, repr(exc))
        finally:
            sim.close()
    return saved


def collect_dagger(
    model,
    demos,
    feature_root: Path,
    index_path: Path,
    scenes: str,
    gpu: int,
    beta: float,
    seed: int,
    iteration: int,
    errors: Path,
    collect_log: Path,
) -> int:
    from spikingnav.vlnce.sim import make_simulator

    done = {row["episode_id"] + "@" + row["scene_id"] for row in _read_index(index_path)}
    pending = [demo for demo in demos if demo["episode_id"] + "@" + demo["scene_id"] not in done]
    print(f"dagger iter {iteration} beta {beta:.6f} pending {len(pending)}", flush=True)
    model.eval()
    saved = 0
    started = time.time()
    _log_collect(collect_log, f"dagger_{iteration:02d}", len(done), 0.0)
    device = next(model.parameters()).device
    for scene_demos in _grouped(pending):
        mesh = mesh_path(scenes, scene_demos[0]["scene_id"])
        if not mesh.is_file():
            _log_error(errors, scene_demos[0], f"missing mesh {mesh}")
            continue
        sim = make_simulator(str(mesh), gpu)
        try:
            for demo in scene_demos:
                destination = _feature_path(feature_root, demo)
                if destination.is_file():
                    continue
                try:
                    episode_rng = random.Random(f"{seed}:{iteration}:{demo['scene_id']}:{demo['episode_id']}")
                    rgb_map, depth_map, language, prev, oracle, _executed = _rollout_dagger(
                        sim, model, demo, beta, episode_rng, device
                    )
                    _save_feature(destination, rgb_map, depth_map, language, prev, oracle)
                    _append_index(index_path, destination, demo, len(oracle))
                    saved += 1
                    if saved % 25 == 0:
                        rate = saved / max(time.time() - started, 1e-6)
                        print(f"dagger {iteration} collected {saved} {rate:.2f} eps/s", flush=True)
                        _log_collect(collect_log, f"dagger_{iteration:02d}", len(done) + saved, rate)
                except Exception as exc:
                    _log_error(errors, demo, repr(exc))
        finally:
            sim.close()
    model.train()
    return saved


def _log_error(path: Path, demo: dict, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"episode_id": demo.get("episode_id"), "scene_id": demo.get("scene_id"), "error": message})
            + "\n"
        )
    print(f"error {demo.get('episode_id')} {message}", flush=True)


class MapEpisodes(Dataset):
    def __init__(self, records: list) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        with np.load(record["path"]) as data:
            return {
                "rgb": torch.from_numpy(np.array(data["rgb"], dtype=np.float32)),
                "depth": torch.from_numpy(np.array(data["depth"], dtype=np.float32)),
                "language": torch.from_numpy(np.array(data["language"], dtype=np.float32)),
                "prev": torch.from_numpy(np.array(data["prev"], dtype=np.int64)),
                "oracle": torch.from_numpy(np.array(data["oracle"], dtype=np.int64)),
            }


def _collate(samples: list) -> dict:
    length = max(int(item["oracle"].shape[0]) for item in samples)
    batch = len(samples)
    rgb = samples[0]["rgb"].new_zeros(length, batch, 768, 3, 3)
    depth = samples[0]["depth"].new_zeros(length, batch, 192, 4, 4)
    language = torch.stack([item["language"] for item in samples])
    prev = torch.full((length, batch), START_ACTION, dtype=torch.long)
    oracle = torch.zeros(length, batch, dtype=torch.long)
    weights = torch.zeros(length, batch)
    for index, item in enumerate(samples):
        steps = int(item["oracle"].shape[0])
        rgb[:steps, index] = item["rgb"]
        depth[:steps, index] = item["depth"]
        prev[:steps, index] = item["prev"]
        oracle[:steps, index] = item["oracle"]
        weights[:steps, index] = inflection_weights(item["oracle"])
    valid = weights > 0
    reset = torch.zeros(length, batch, dtype=torch.bool)
    reset[0] = True
    detach = torch.zeros(length, batch, dtype=torch.bool)
    return {
        "rgb": rgb,
        "depth": depth,
        "language": language,
        "prev": prev,
        "oracle": oracle,
        "weights": weights,
        "valid": valid,
        "reset": reset,
        "detach": detach,
    }


def _batches(records: list, batch_size: int, seed: int, epoch: int) -> list:
    order = list(range(len(records)))
    rng = random.Random(seed + epoch * 17)
    rng.shuffle(order)
    usable = len(order) - (len(order) % batch_size)
    order = order[:usable]
    preload = batch_size * 22
    batches = []
    for start in range(0, usable, preload):
        chunk = order[start : start + preload]
        chunk.sort(key=lambda item: records[item]["steps"])
        width = len(chunk) - (len(chunk) % batch_size)
        for offset in range(0, width, batch_size):
            batches.append(chunk[offset : offset + batch_size])
    rng.shuffle(batches)
    return batches


def _move(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def train_epochs(
    model,
    optimizer,
    records: list,
    output: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
    stage: str,
    dagger_it: int,
    stem: str,
    max_steps: int,
    log_path: Path,
    global_step: int,
) -> int:
    finished = sorted(int(path.stem.split("_")[-1]) for path in output.glob(f"{stem}_*.pt"))
    start = finished[-1] + 1 if finished else 0
    if finished:
        load_checkpoint(output / f"{stem}_{finished[-1]:02d}.pt", model, optimizer)
        model.to(device)
    if start >= epochs:
        print(f"{stage} already has {epochs} epochs", flush=True)
        return global_step
    if len(records) < batch_size:
        raise RuntimeError(f"{stage} has {len(records)} episodes, batch size is {batch_size}")
    dataset = MapEpisodes(records)
    for epoch in range(start, epochs):
        model.train()
        order = _batches(records, batch_size, seed, epoch)
        totals = {"loss": 0.0, "correct": 0.0, "count": 0.0}
        started = time.time()
        for step, indexes in enumerate(order):
            if max_steps and step >= max_steps:
                break
            samples = [dataset[index] for index in indexes]
            batch = _move(_collate(samples), device)
            optimizer.zero_grad(set_to_none=True)
            logits, _state, _aux = model.forward_from_maps(
                batch["rgb"],
                batch["depth"],
                batch["language"],
                batch["prev"],
                batch["valid"],
                batch["reset"],
                batch["detach"],
            )
            loss = weighted_action_loss(logits, batch["oracle"], batch["weights"])
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at {stage} epoch {epoch} step {step}")
            loss.backward()
            optimizer.step()
            supervised = batch["weights"] > 0
            chosen = logits.detach().argmax(dim=-1)
            correct = int(((chosen == batch["oracle"]) & supervised).sum().item())
            count = int(supervised.sum().item())
            totals["loss"] += float(loss.detach()) * count
            totals["correct"] += correct
            totals["count"] += count
            _append_row(
                log_path,
                {
                    "step": global_step,
                    "loss": float(loss.detach()),
                    "accuracy": correct / count if count else 0.0,
                    "supervised_steps": count,
                    "epoch": epoch,
                    "stage": stage,
                    "dagger_iteration": dagger_it,
                },
            )
            global_step += 1
            if step % 20 == 0:
                print(
                    f"{stage} epoch {epoch} step {step}/{len(order)} loss {float(loss.detach()):.4f}",
                    flush=True,
                )
        if max_steps:
            print(f"{stage} epoch {epoch} stopped after {max_steps} steps for the short run", flush=True)
            return global_step
        count = max(totals["count"], 1.0)
        record = {
            "stage": stage,
            "epoch": epoch,
            "dagger_iteration": dagger_it,
            "loss": totals["loss"] / count,
            "accuracy": totals["correct"] / count,
            "supervised_steps": int(totals["count"]),
            "batches": len(order),
            "seconds": time.time() - started,
        }
        save_checkpoint(output / f"{stem}_{epoch:02d}.pt", model, optimizer, epoch, record)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
    return global_step


def _load_init(path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu")
    raw = dict(payload["config"])
    populations = PopulationSizes(**raw.pop("populations"))
    raw.pop("neuron_count", None)
    model = BrainVLNModel(BrainVLNConfig(populations=populations, **raw))
    model.load_state_dict(payload["model"])
    model.to(device)
    return model


def _optimizer(model, lr: float):
    return torch.optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad), lr=lr)


_ACTIVE_COLLECTORS: list = []


def _parse_gpus(spec: str, fallback: int) -> list:
    if spec.strip():
        gpus = [int(part) for part in spec.split(",") if part.strip()]
    else:
        gpus = [int(fallback)]
    if not gpus or any(gpu < 0 for gpu in gpus):
        raise ValueError(f"invalid gpus {spec!r}")
    return gpus


def _worker_slots(gpus: list, workers_per_gpu: int) -> list:
    if workers_per_gpu < 1:
        raise ValueError("workers-per-gpu must be positive")
    return [(int(gpu), replica) for gpu in gpus for replica in range(workers_per_gpu)]


def _count_npz(root: Path) -> int:
    if not root.is_dir():
        return 0
    return sum(1 for path in root.rglob("*.npz") if not path.name.endswith(".tmp.npz"))


def _policy_checkpoint(output: Path, iteration: int, init_path: Path) -> Path:
    if iteration == 0:
        envdrop = sorted(output.glob("ckpt_envdrop_*.pt"))
        return envdrop[-1] if envdrop else init_path
    previous = sorted(output.glob(f"ckpt_dagger_{iteration - 1:02d}_*.pt"))
    if not previous:
        raise RuntimeError(f"iteration {iteration} needs the checkpoint from iteration {iteration - 1}")
    return previous[-1]


def _stop_collectors() -> None:
    for proc in list(_ACTIVE_COLLECTORS):
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def _handle_stop(signum, _frame) -> None:
    _stop_collectors()
    raise SystemExit(128 + int(signum))


def _merge_shard_indexes(output: Path, iteration: int, num_shards: int, feature_root: Path) -> list:
    index_path = output / f"dagger_index_{iteration:02d}.jsonl"
    rows = _read_index(index_path)
    known = {f"{row['episode_id']}@{row['scene_id']}" for row in rows}
    for shard in range(num_shards):
        shard_path = output / "shards" / f"dagger_index_{iteration:02d}_shard{shard:02d}.jsonl"
        for row in _read_index(shard_path):
            key = f"{row['episode_id']}@{row['scene_id']}"
            if key in known:
                continue
            known.add(key)
            _append_index(index_path, Path(row["path"]), row, int(row["steps"]))
    return _reconcile_index(index_path, feature_root)


def _start_collector(
    args,
    output: Path,
    iteration: int,
    shard: int,
    gpu: int,
    num_shards: int,
    init_path: Path,
    stage: str = "dagger-collect",
    episode_part: int = 0,
    episode_parts: int = 1,
):
    repo = Path(__file__).resolve().parents[2]
    log_dir = output / "shards"
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--init",
        str(init_path),
        "--output",
        str(output),
        "--dataset",
        args.dataset,
        "--scenes",
        args.scenes,
        "--gpu",
        str(gpu),
        "--seed",
        str(args.seed),
        "--limit",
        str(args.limit),
        "--stage",
        stage,
        "--iteration",
        str(iteration),
        "--shard",
        str(shard),
        "--num-shards",
        str(num_shards),
        "--episode-part",
        str(episode_part),
        "--episode-parts",
        str(episode_parts),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    env["OPENBLAS_NUM_THREADS"] = "4"
    env["NUMEXPR_NUM_THREADS"] = "4"
    tag = "envdrop" if stage == "envdrop-collect" else f"iter{iteration:02d}"
    part_tag = f"_part{episode_part:02d}of{episode_parts:02d}" if episode_parts > 1 else ""
    log_path = log_dir / f"collect_{tag}_shard{shard:02d}{part_tag}.stdout"
    handle = log_path.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    _ACTIVE_COLLECTORS.append(proc)
    print(f"collector shard {shard} gpu {gpu} pid {proc.pid}", flush=True)
    return proc, handle, log_path


def _collect_sharded(args, output: Path, iteration: int, chosen: list, feature_root: Path, slots: list, init_path: Path) -> None:
    num_shards = len(slots)
    work = []
    for shard, (gpu, _replica) in enumerate(slots):
        mine = shard_dagger_demos(chosen, num_shards, shard)
        left = [demo for demo in mine if not _feature_path(feature_root, demo).is_file()]
        if not left:
            print(f"collector shard {shard} gpu {gpu} already complete ({len(mine)} episodes)", flush=True)
            continue
        print(f"collector shard {shard} gpu {gpu} pending {len(left)} of {len(mine)}", flush=True)
        work.append((shard, gpu))
    if not work:
        return
    attempts = {shard: 0 for shard, _gpu in work}
    active = []
    for shard, gpu in work:
        attempts[shard] += 1
        proc, handle, log_path = _start_collector(args, output, iteration, shard, gpu, num_shards, init_path)
        active.append((proc, handle, log_path, shard, gpu))
        time.sleep(2)
    started = time.time()
    start_count = _count_npz(feature_root)
    last_logged = start_count
    last_print = started
    collect_log = output / "collect_log.jsonl"
    while active:
        time.sleep(5)
        still = []
        for proc, handle, log_path, shard, gpu in active:
            code = proc.poll()
            if code is None:
                still.append((proc, handle, log_path, shard, gpu))
                continue
            handle.close()
            if proc in _ACTIVE_COLLECTORS:
                _ACTIVE_COLLECTORS.remove(proc)
            if code != 0:
                tail = ""
                if log_path.is_file():
                    tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:])
                print(f"collector shard {shard} gpu {gpu} exited {code}\n{tail}", flush=True)
                if attempts[shard] < 2:
                    attempts[shard] += 1
                    print(f"retry collector shard {shard} gpu {gpu}", flush=True)
                    time.sleep(2)
                    proc, handle, log_path = _start_collector(
                        args, output, iteration, shard, gpu, num_shards, init_path
                    )
                    still.append((proc, handle, log_path, shard, gpu))
                else:
                    _stop_collectors()
                    raise RuntimeError(f"collector shard {shard} on gpu {gpu} failed twice")
            else:
                print(f"collector shard {shard} gpu {gpu} finished", flush=True)
        active = still
        current = _count_npz(feature_root)
        now = time.time()
        rate = (current - start_count) / max(now - started, 1e-6)
        if current >= last_logged + 25:
            _log_collect(collect_log, f"dagger_{iteration:02d}", current, rate)
            last_logged = current
        if now - last_print >= 30:
            print(
                f"dagger {iteration} collected {current} {rate:.3f} eps/s alive {len(active)}",
                flush=True,
            )
            last_print = now
    current = _count_npz(feature_root)
    rate = (current - start_count) / max(time.time() - started, 1e-6)
    _log_collect(collect_log, f"dagger_{iteration:02d}", current, rate)
    print(f"dagger {iteration} collection pass finished at {current} episodes {rate:.3f} eps/s", flush=True)


def _collect_envdrop_sharded(args, output: Path, demos: list, feature_root: Path, slots: list, init_path: Path) -> None:
    num_shards = len(slots)
    work = []
    for shard, (gpu, _replica) in enumerate(slots):
        mine = shard_dagger_demos(demos, num_shards, shard)
        left = [demo for demo in mine if not _feature_path(feature_root, demo).is_file()]
        if not left:
            print(f"envdrop shard {shard} gpu {gpu} already complete ({len(mine)} episodes)", flush=True)
            continue
        print(f"envdrop shard {shard} gpu {gpu} pending {len(left)} of {len(mine)}", flush=True)
        work.append((shard, gpu, len(left)))
    if not work:
        return
    launches = []
    biggest = max(work, key=lambda item: item[2])
    others = [item for item in work if item[0] != biggest[0]]
    other_left = sum(item[2] for item in others)
    if len(slots) > 1 and biggest[2] >= 1000 and biggest[2] > other_left:
        shard = biggest[0]
        parts = len(slots)
        print(
            f"envdrop shard {shard} has {biggest[2]} episodes left; split across {parts} workers",
            flush=True,
        )
        for part, (gpu, _replica) in enumerate(slots):
            launches.append((shard, gpu, part, parts))
        for shard, gpu, _left in others:
            launches.append((shard, gpu, 0, 1))
    else:
        for shard, gpu, _left in work:
            launches.append((shard, gpu, 0, 1))
    attempts = {(shard, part): 0 for shard, _gpu, part, _parts in launches}
    active = []
    for shard, gpu, part, parts in launches:
        attempts[(shard, part)] += 1
        proc, handle, log_path = _start_collector(
            args,
            output,
            0,
            shard,
            gpu,
            num_shards,
            init_path,
            stage="envdrop-collect",
            episode_part=part,
            episode_parts=parts,
        )
        active.append((proc, handle, log_path, shard, gpu, part, parts))
        time.sleep(2)
    started = time.time()
    start_count = _count_npz(feature_root)
    last_logged = start_count
    last_print = started
    collect_log = output / "collect_log.jsonl"
    while active:
        time.sleep(5)
        still = []
        for proc, handle, log_path, shard, gpu, part, parts in active:
            code = proc.poll()
            if code is None:
                still.append((proc, handle, log_path, shard, gpu, part, parts))
                continue
            handle.close()
            if proc in _ACTIVE_COLLECTORS:
                _ACTIVE_COLLECTORS.remove(proc)
            if code != 0:
                tail = ""
                if log_path.is_file():
                    tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:])
                print(f"envdrop shard {shard} part {part} gpu {gpu} exited {code}\n{tail}", flush=True)
                key = (shard, part)
                if attempts[key] < 2:
                    attempts[key] += 1
                    print(f"retry envdrop shard {shard} part {part} gpu {gpu}", flush=True)
                    time.sleep(2)
                    proc, handle, log_path = _start_collector(
                        args,
                        output,
                        0,
                        shard,
                        gpu,
                        num_shards,
                        init_path,
                        stage="envdrop-collect",
                        episode_part=part,
                        episode_parts=parts,
                    )
                    still.append((proc, handle, log_path, shard, gpu, part, parts))
                else:
                    _stop_collectors()
                    raise RuntimeError(f"envdrop shard {shard} part {part} on gpu {gpu} failed twice")
            else:
                print(f"envdrop shard {shard} part {part} gpu {gpu} finished", flush=True)
        active = still
        current = _count_npz(feature_root)
        now = time.time()
        rate = (current - start_count) / max(now - started, 1e-6)
        if current >= last_logged + 25:
            _log_collect(collect_log, "envdrop", current, rate)
            last_logged = current
        if now - last_print >= 30:
            print(
                f"envdrop collected {current} {rate:.3f} eps/s alive {len(active)}",
                flush=True,
            )
            last_print = now
    current = _count_npz(feature_root)
    rate = (current - start_count) / max(time.time() - started, 1e-6)
    _log_collect(collect_log, "envdrop", current, rate)
    print(f"envdrop collection pass finished at {current} episodes {rate:.3f} eps/s", flush=True)


def _merge_envdrop_indexes(output: Path, num_shards: int, feature_root: Path) -> list:
    index_path = output / "envdrop_index.jsonl"
    rows = _read_index(index_path)
    known = {f"{row['episode_id']}@{row['scene_id']}" for row in rows}
    for shard in range(num_shards):
        shard_path = output / "shards" / f"envdrop_index_shard{shard:02d}.jsonl"
        for row in _read_index(shard_path):
            key = f"{row['episode_id']}@{row['scene_id']}"
            if key in known:
                continue
            known.add(key)
            _append_index(index_path, Path(row["path"]), row, int(row["steps"]))
    return _reconcile_index(index_path, feature_root)


def _envdrop_collect_worker(args) -> int:
    if args.num_shards < 1 or not 0 <= args.shard < args.num_shards:
        raise SystemExit("envdrop-collect needs --shard and --num-shards")
    try:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    except RuntimeError:
        pass
    output = Path(args.output)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
    model = _load_init(Path(args.init), device)
    demos = _cached_demos(output / "envdrop_episodes.pkl", Path(args.dataset), "joint_train_envdrop", True)
    if args.limit > 0:
        demos = demos[: args.limit]
    mine = partition_episodes(
        shard_dagger_demos(demos, args.num_shards, args.shard),
        args.episode_part,
        args.episode_parts,
    )
    feature_root = output / "envdrop_features"
    part_tag = f"_part{args.episode_part:02d}" if args.episode_parts > 1 else ""
    index_path = output / "shards" / f"envdrop_index_shard{args.shard:02d}{part_tag}.jsonl"
    print(
        f"envdrop shard {args.shard}/{args.num_shards} part {args.episode_part}/{args.episode_parts} "
        f"gpu {args.gpu} episodes {len(mine)}",
        flush=True,
    )
    collect_teacher(
        model,
        mine,
        feature_root,
        index_path,
        args.scenes,
        args.gpu,
        output / "shards" / f"envdrop_errors_shard{args.shard:02d}{part_tag}.jsonl",
        output / "shards" / f"envdrop_collect_shard{args.shard:02d}{part_tag}.jsonl",
    )
    return 0


def _dagger_collect_worker(args) -> int:
    if args.iteration < 0 or args.num_shards < 1 or not 0 <= args.shard < args.num_shards:
        raise SystemExit("dagger-collect needs --iteration, --shard, and --num-shards")
    try:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    except RuntimeError:
        pass
    output = Path(args.output)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
    model = _load_init(Path(args.init), device)
    train_demos = _cached_demos(output / "train_episodes.pkl", Path(args.dataset), "train", False)
    chosen = select_dagger_episodes(train_demos, args.iteration, args.seed, args.limit)
    mine = shard_dagger_demos(chosen, args.num_shards, args.shard)
    feature_root = output / "dagger_features" / f"iter_{args.iteration:02d}"
    index_path = output / "shards" / f"dagger_index_{args.iteration:02d}_shard{args.shard:02d}.jsonl"
    beta = dagger_beta(DAGGER_P, args.iteration + 1)
    print(
        f"shard {args.shard}/{args.num_shards} gpu {args.gpu} episodes {len(mine)} beta {beta:.6f}",
        flush=True,
    )
    collect_dagger(
        model,
        mine,
        feature_root,
        index_path,
        args.scenes,
        args.gpu,
        beta,
        args.seed,
        args.iteration,
        output / "shards" / f"errors_iter{args.iteration:02d}_shard{args.shard:02d}.jsonl",
        output / "shards" / f"collect_iter{args.iteration:02d}_shard{args.shard:02d}.jsonl",
    )
    return 0


def _ensure_model(model, optimizer, init_path: Path, device: torch.device):
    if model is None:
        model = _load_init(init_path, device)
        if model.frontend.pretrained is None:
            raise RuntimeError("this schedule needs the frozen MLANet encoder")
        optimizer = _optimizer(model, ENVDROP_LR)
    else:
        model.to(device)
    return model, optimizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--dataset",
        default="/home/vln/dhj/DPedPro-VLN/data/datasets/R2R_VLNCE_FSASub",
    )
    parser.add_argument("--scenes", default="data/scene_datasets")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpus", default="", help="Comma-separated GPU ids. Empty uses --gpu.")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--episode-part", type=int, default=0)
    parser.add_argument("--episode-parts", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="Collect at most this many episodes per stage. 0 means the MLANet count.")
    parser.add_argument("--max-train-steps", type=int, default=0, help="Stop each epoch after this many optimizer steps. 0 trains the full epoch.")
    parser.add_argument(
        "--stage",
        choices=("all", "envdrop", "envdrop-collect", "dagger", "dagger-collect"),
        default="all",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if args.workers_per_gpu < 1:
        parser.error("workers-per-gpu must be positive")
    if args.episode_parts < 1 or not 0 <= args.episode_part < args.episode_parts:
        parser.error("episode-part must be inside episode-parts")
    gpus = _parse_gpus(args.gpus, args.gpu)
    args.gpu = gpus[0]
    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)
    if args.stage == "dagger-collect":
        return _dagger_collect_worker(args)
    if args.stage == "envdrop-collect":
        return _envdrop_collect_worker(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    dataset_root = Path(args.dataset)
    init_path = Path(args.init)
    slots = _worker_slots(gpus, args.workers_per_gpu)
    sharded = args.stage in ("dagger", "envdrop", "all") and len(slots) > 1
    if sharded:
        model = None
        optimizer = None
    else:
        model = _load_init(init_path, device)
        if model.frontend.pretrained is None:
            raise RuntimeError("this schedule needs the frozen MLANet encoder")
        optimizer = _optimizer(model, ENVDROP_LR)
    envdrop_root = output / "envdrop_features"
    dagger_root = output / "dagger_features"
    errors = output / "errors.jsonl"
    log_path = output / "train_log.jsonl"
    limit = args.limit

    def take(demos, count):
        chosen = demos if limit <= 0 else demos[:limit]
        if count > 0:
            chosen = chosen[:count]
        return chosen

    provenance = {
        "init_checkpoint": str(init_path),
        "init_val_unseen_sr": 0.1566068515497553,
        "init_val_unseen_update": 9000,
        "init_checkpoint_name": "ckpt_epoch_89.pt",
        "algorithm": (
            "MLANet DAgger from the v2 checkpoint, EnvDrop skipped"
            if args.stage == "dagger"
            else "MLANet EnvDrop teacher forcing, then DAgger"
        ),
        "envdrop_config": "third_party/MLANet/mlanet/config/mla/mla_aug.yaml",
        "dagger_config": "third_party/MLANet/mlanet/config/mla/mla_aug_da_tune.yaml",
        "envdrop_epochs": ENVDROP_EPOCHS,
        "envdrop_p": ENVDROP_P,
        "envdrop_update_size_published": ENVDROP_UPDATE_SIZE,
        "dagger_epochs": DAGGER_EPOCHS,
        "dagger_iterations": DAGGER_ITERATIONS,
        "dagger_update_size": DAGGER_UPDATE_SIZE,
        "dagger_p": DAGGER_P,
        "dagger_beta": "p ** (iteration + 1) because load_from_ckpt is true",
        "lr": ENVDROP_LR,
        "inflection_weight_coef": INFLECTION_COEF,
        "optimizer": "Adam",
        "grad_clip": None,
        "mlanet_batch_size": 5,
        "batch_size": args.batch_size,
        "batch_size_note": (
            "MLANet uses 5 trajectories per optimizer step. This run uses a wider "
            "batch so the reservoir keeps the GPU busy. Epochs, learning rate, "
            "inflection-weighted loss, DAgger iterations, update size, and beta stay "
            "the published values."
        ),
        "loss": "mean over trajectories of inflection-weighted expert-action cross-entropy",
        "depth_note": (
            "Observations are rendered at 224 and the frozen depth encoder resizes "
            "them to 256, matching BrainVLN_v2. MLANet's DAgger yaml renders depth at 256."
        ),
        "expert": "habitat_sim.GreedyGeodesicFollower goal_radius 0.5 during DAgger; EnvDrop replays stored GT actions",
        "dataset": str(dataset_root),
        "scenes": args.scenes,
        "seed": args.seed,
        "limit": limit,
        "gpus": gpus,
        "workers_per_gpu": args.workers_per_gpu,
        "collection": (
            "scene-sharded habitat replicas, one process per worker, threads capped at 4"
            if sharded
            else "one habitat simulator"
        ),
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "head": _git(["rev-parse", "HEAD"]),
        "status_short": _git(["status", "--short"]),
        "source_sha256": _source_hash(),
        "command": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    previous = output / "provenance.json"
    archive = output / "provenance_single_gpu.json"
    if previous.is_file() and not archive.is_file():
        archive.write_text(previous.read_text(encoding="utf-8"), encoding="utf-8")
    previous.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    global_step = _next_train_step(log_path)
    if args.stage in ("all", "envdrop"):
        demos = _cached_demos(output / "envdrop_episodes.pkl", dataset_root, "joint_train_envdrop", True)
        print(f"envdrop episodes {len(demos)} published update_size {ENVDROP_UPDATE_SIZE}", flush=True)
        selected = take(demos, 0)
        collect_log = output / "collect_log.jsonl"
        _reconcile_index(output / "envdrop_index.jsonl", envdrop_root)
        if len(slots) > 1:
            if model is not None:
                model.cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                model = None
            _collect_envdrop_sharded(args, output, selected, envdrop_root, slots, init_path)
            records = _merge_envdrop_indexes(output, len(slots), envdrop_root)
            model, optimizer = _ensure_model(model, optimizer, init_path, device)
        else:
            collect_teacher(
                model,
                selected,
                envdrop_root,
                output / "envdrop_index.jsonl",
                args.scenes,
                args.gpu,
                errors,
                collect_log,
            )
            records = _reconcile_index(output / "envdrop_index.jsonl", envdrop_root)
        global_step = train_epochs(
            model,
            optimizer,
            records,
            output,
            device,
            ENVDROP_EPOCHS,
            args.batch_size,
            args.seed,
            "envdrop",
            0,
            "ckpt_envdrop",
            args.max_train_steps,
            log_path,
            global_step,
        )
        if args.max_train_steps:
            print("short run finished during envdrop; dagger was not started", flush=True)
            return 0

    if args.stage in ("all", "dagger"):
        envdrop_ckpts = sorted(output.glob("ckpt_envdrop_*.pt"))
        if envdrop_ckpts:
            model, optimizer = _ensure_model(model, optimizer, init_path, device)
            load_checkpoint(envdrop_ckpts[-1], model, None)
            model.to(device)
            optimizer = _optimizer(model, ENVDROP_LR)
        elif args.stage != "dagger":
            raise RuntimeError("DAgger needs a finished EnvDrop checkpoint in the output directory")
        train_demos = _cached_demos(output / "train_episodes.pkl", dataset_root, "train", False)
        print(f"dagger source train episodes {len(train_demos)}", flush=True)
        accumulated = []
        for iteration in range(DAGGER_ITERATIONS):
            chosen = select_dagger_episodes(train_demos, iteration, args.seed, limit)
            beta = dagger_beta(DAGGER_P, iteration + 1)
            feature_root = dagger_root / f"iter_{iteration:02d}"
            index_path = output / f"dagger_index_{iteration:02d}.jsonl"
            print(f"dagger iter {iteration} beta {beta:.6f} episodes {len(chosen)}", flush=True)
            if sharded:
                if model is not None:
                    model.cpu()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                _collect_sharded(
                    args,
                    output,
                    iteration,
                    chosen,
                    feature_root,
                    slots,
                    _policy_checkpoint(output, iteration, init_path),
                )
                records = _merge_shard_indexes(output, iteration, len(slots), feature_root)
                model, optimizer = _ensure_model(model, optimizer, init_path, device)
            else:
                collect_dagger(
                    model,
                    chosen,
                    feature_root,
                    index_path,
                    args.scenes,
                    args.gpu,
                    beta,
                    args.seed,
                    iteration,
                    errors,
                    output / "collect_log.jsonl",
                )
                records = _reconcile_index(index_path, feature_root)
            accumulated.extend(records)
            global_step = train_epochs(
                model,
                optimizer,
                accumulated,
                output,
                device,
                DAGGER_EPOCHS,
                args.batch_size,
                args.seed + 1000 + iteration,
                f"dagger_{iteration:02d}",
                iteration,
                f"ckpt_dagger_{iteration:02d}",
                args.max_train_steps,
                log_path,
                global_step,
            )
            if args.max_train_steps:
                print("short run finished during dagger", flush=True)
                return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
