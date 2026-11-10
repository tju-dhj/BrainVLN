"""Habitat-Sim replay of R2R-CE expert actions, and closed-loop evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from spikingnav.vlnce.constants import (
    HFOV,
    MAX_EPISODE_STEPS,
    MOVE_FORWARD,
    RGB_SIZE,
    SIM_ACTIONS,
    STOP,
    SUCCESS_DISTANCE,
    TURN_LEFT,
    TURN_RIGHT,
)
from spikingnav.vlnce.episodes import mesh_path, token_length
from spikingnav.vlnce.metrics import ndtw, path_length, spl
from spikingnav.vlnce.store import READY_NAME, episode_dir, save_rollout, scene_name


def _habitat():
    try:
        import habitat_sim
    except ImportError as exc:
        raise ImportError(
            "habitat_sim is required to replay or evaluate R2R-CE. "
            "The project spikingnav environment does not include it; "
            "on this machine Habitat-Sim 0.3.x is required; use the habnav environment on this host."
        ) from exc
    return habitat_sim


def rotation_from_coeffs(coeffs) -> object:
    habitat_sim = _habitat()
    helper = getattr(getattr(habitat_sim, "utils", None), "common", None)
    if helper is not None and hasattr(helper, "quat_from_coeffs"):
        return helper.quat_from_coeffs(np.asarray(coeffs, dtype=np.float32))
    import quaternion

    values = np.asarray(coeffs, dtype=np.float64)
    return quaternion.quaternion(values[3], values[0], values[1], values[2])


def make_simulator(scene_file: str, gpu_device_id: int = 0):
    habitat_sim = _habitat()
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(scene_file)
    sim_cfg.gpu_device_id = int(gpu_device_id)
    sim_cfg.allow_sliding = True
    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb"
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [RGB_SIZE, RGB_SIZE]
    rgb.hfov = 90
    rgb.position = [0.0, 1.25, 0.0]
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height = 1.5
    agent_cfg.radius = 0.1
    depth = habitat_sim.CameraSensorSpec()
    depth.uuid = "depth"
    depth.sensor_type = habitat_sim.SensorType.DEPTH
    depth.resolution = [RGB_SIZE, RGB_SIZE]
    depth.hfov = HFOV
    depth.position = [0.0, 1.25, 0.0]
    agent_cfg.sensor_specifications = [rgb, depth]
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=15.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=15.0)
        ),
    }
    simulator = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    navmesh = Path(scene_file).with_suffix(".navmesh")
    if navmesh.is_file():
        simulator.pathfinder.load_nav_mesh(str(navmesh))
    return simulator


def _place(sim, position, rotation) -> None:
    habitat_sim = _habitat()
    state = habitat_sim.AgentState()
    state.position = np.asarray(position, dtype=np.float32)
    state.rotation = rotation_from_coeffs(rotation)
    sim.get_agent(0).set_state(state)


def _rgb(sim) -> np.ndarray:
    frame = np.asarray(sim.get_sensor_observations()["rgb"])
    return np.ascontiguousarray(frame[..., :3])

def _depth(sim) -> np.ndarray:
    return np.asarray(sim.get_sensor_observations()["depth"], dtype=np.float32).copy()


def _position(sim) -> np.ndarray:
    return np.asarray(sim.get_agent(0).get_state().position, dtype=np.float32).copy()


def replay_expert(sim, demo: dict) -> dict:
    """Step the stored expert actions from the episode start pose."""
    _place(sim, demo["start_position"], demo["start_rotation"])
    frames = []
    depths = []
    positions = []
    actions = []
    for action in demo["actions"]:
        action = int(action)
        if action not in (STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT):
            raise ValueError(f"Unknown R2R-CE action {action}")
        frames.append(_rgb(sim))
        depths.append(_depth(sim))
        positions.append(_position(sim))
        actions.append(action)
        if action == STOP:
            break
        sim.step(SIM_ACTIONS[action])
    record = dict(demo)
    record["rgb"] = frames
    record["depth"] = depths
    record["positions"] = positions
    record["actions"] = actions
    return record


def geodesic(sim, position, goal) -> float:
    habitat_sim = _habitat()
    try:
        pathfinder = sim.pathfinder
        start = pathfinder.snap_point(np.asarray(position, dtype=np.float32))
        end = pathfinder.snap_point(np.asarray(goal, dtype=np.float32))
        if not pathfinder.is_navigable(start) or not pathfinder.is_navigable(end):
            return float("inf")
        query = habitat_sim.ShortestPath()
        query.requested_start = start
        query.requested_end = end
        if not pathfinder.find_path(query):
            return float("inf")
        return float(query.geodesic_distance)
    except Exception:
        return float("inf")


def shard_scenes(grouped: Dict[str, List[dict]], num_shards: int, shard_index: int) -> List[str]:
    """Balance scenes across workers by episode count. Each scene stays on one worker."""
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError(f"invalid shard {shard_index}/{num_shards}")
    loads = [0] * num_shards
    buckets: List[List[str]] = [[] for _ in range(num_shards)]
    for scene_id in sorted(grouped, key=lambda key: (-len(grouped[key]), key)):
        index = min(range(num_shards), key=lambda item: (loads[item], item))
        buckets[index].append(scene_id)
        loads[index] += len(grouped[scene_id])
    return buckets[shard_index]


def collect_split(
    demos: List[dict],
    scenes_dir: str,
    output_dir: str,
    gpu_device_id: int = 0,
    split: str = "train",
    shard_index: int = 0,
    num_shards: int = 1,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    grouped: Dict[str, List[dict]] = {}
    for demo in demos:
        grouped.setdefault(demo["scene_id"], []).append(demo)
    selected = shard_scenes(grouped, num_shards, shard_index)
    saved = 0
    skipped = 0
    errors = []
    log_dir = output / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest = log_dir / f"manifest_shard{shard_index}.jsonl"
    error_path = log_dir / f"errors_shard{shard_index}.jsonl"
    print(
        f"shard {shard_index}/{num_shards} scenes {len(selected)} "
        f"episodes {sum(len(grouped[scene]) for scene in selected)}",
        flush=True,
    )
    for scene_id in selected:
        scene_demos = grouped[scene_id]
        mesh = mesh_path(scenes_dir, scene_id)
        pending = []
        for demo in scene_demos:
            destination = episode_dir(str(output), split, scene_id, demo["episode_id"])
            if (destination / READY_NAME).is_file():
                skipped += 1
                continue
            pending.append(demo)
        if not pending:
            print(f"scene {scene_name(scene_id)} already complete ({len(scene_demos)})", flush=True)
            continue
        if not mesh.is_file():
            for demo in pending:
                errors.append({"episode_id": demo["episode_id"], "error": f"missing mesh {mesh}"})
            continue
        try:
            sim = make_simulator(str(mesh), gpu_device_id)
        except Exception as exc:
            for demo in pending:
                errors.append({"episode_id": demo["episode_id"], "error": str(exc)})
            continue
        print(f"scene {scene_name(scene_id)} pending {len(pending)}", flush=True)
        try:
            for demo in pending:
                try:
                    record = replay_expert(sim, demo)
                    destination = save_rollout(str(output), record, split=split)
                    with manifest.open("a") as handle:
                        handle.write(json.dumps({
                            "episode_id": demo["episode_id"],
                            "scene_id": scene_id,
                            "path": str(destination),
                            "steps": len(record["actions"]),
                            "geodesic_distance": demo["geodesic_distance"],
                        }) + "\n")
                    saved += 1
                    if saved % 25 == 0:
                        print(f"shard {shard_index} saved {saved} skipped {skipped}", flush=True)
                except Exception as exc:
                    errors.append({"episode_id": demo["episode_id"], "error": str(exc)})
                    print(f"error {demo['episode_id']} {exc}", flush=True)
        finally:
            sim.close()
    if errors:
        error_path.write_text("".join(json.dumps(row) + "\n" for row in errors))
    summary = {
        "saved": saved,
        "skipped": skipped,
        "errors": len(errors),
        "split": split,
        "shard": shard_index,
        "num_shards": num_shards,
        "output": str(output),
    }
    (log_dir / f"summary_shard{shard_index}.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def evaluate(
    demos: List[dict],
    scenes_dir: str,
    checkpoint: str,
    output_dir: str,
    gpu_device_id: int = 0,
    max_steps: int = MAX_EPISODE_STEPS,
) -> dict:
    import torch

    from spikingnav.vlnce.il import load_policy

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(checkpoint, device)
    grouped: Dict[str, List[dict]] = {}
    for demo in demos:
        grouped.setdefault(demo["scene_id"], []).append(demo)
    rows = []
    for scene_id, scene_demos in grouped.items():
        mesh = mesh_path(scenes_dir, scene_id)
        if not mesh.is_file():
            for demo in scene_demos:
                rows.append({"episode_id": demo["episode_id"], "error": f"missing mesh {mesh}"})
            continue
        sim = make_simulator(str(mesh), gpu_device_id)
        try:
            for demo in scene_demos:
                try:
                    rows.append(_rollout_policy(sim, policy, demo, device, max_steps))
                except Exception as exc:
                    rows.append({"episode_id": demo["episode_id"], "error": str(exc)})
        finally:
            sim.close()
    scored = [row for row in rows if "success" in row]
    summary = {
        "episodes": len(rows),
        "scored": len(scored),
        "errors": len(rows) - len(scored),
        "sr": _mean(scored, "success"),
        "spl": _mean(scored, "spl"),
        "ne": _mean(scored, "navigation_error"),
        "ndtw": _mean(scored, "ndtw"),
        "oracle_sr": _mean(scored, "oracle_success"),
    }
    (output / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _mean(rows: List[dict], key: str):
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        values.append(float(value))
    if not values:
        return None
    return float(sum(values) / len(values))


def _rollout_policy(sim, policy, demo: dict, device, max_steps: int) -> dict:
    import torch

    from spikingnav.vlnce.rgb import prepare_rgb

    _place(sim, demo["start_position"], demo["start_rotation"])
    tokens = torch.as_tensor(demo["tokens"], dtype=torch.long, device=device).unsqueeze(0)
    lengths = torch.tensor([token_length(demo["tokens"])], device=device)
    memory = policy.initial_memory(1, device)
    positions = []
    distances = []
    stopped = False
    for step in range(max_steps):
        positions.append(_position(sim))
        distances.append(geodesic(sim, positions[-1], demo["goal"]))
        frame = torch.from_numpy(_rgb(sim)).unsqueeze(0).unsqueeze(0)
        rgb = prepare_rgb(frame).to(device)
        mask = rgb.new_zeros(1, 1, 1) if step == 0 else rgb.new_ones(1, 1, 1)
        with torch.no_grad():
            action, memory = policy.act(rgb, tokens, lengths, memory, mask)
        choice = int(action.item())
        if choice == STOP:
            stopped = True
            break
        if choice not in SIM_ACTIONS:
            break
        sim.step(SIM_ACTIONS[choice])
    final = distances[-1] if distances else float("inf")
    success = bool(stopped and np.isfinite(final) and final <= SUCCESS_DISTANCE)
    traveled = path_length(positions)
    geodesic_start = demo["geodesic_distance"]
    if geodesic_start < 0 and positions:
        geodesic_start = geodesic(sim, positions[0], demo["goal"])
    reference = demo["locations"] or demo["reference_path"]
    oracle = bool(any(np.isfinite(d) and d <= SUCCESS_DISTANCE for d in distances))
    return {
        "episode_id": demo["episode_id"],
        "scene_id": demo["scene_id"],
        "success": success,
        "oracle_success": oracle,
        "spl": spl(success, geodesic_start, traveled),
        "navigation_error": None if not np.isfinite(final) else float(final),
        "ndtw": ndtw(positions, reference),
        "steps": len(positions),
        "stopped": stopped,
        "path_length": traveled,
    }
