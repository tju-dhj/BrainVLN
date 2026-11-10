"""Read-only structural audit for collected RGB-D R2R-CE episodes."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

EXPECTED_ACTION_NAMES = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
FRAME_PATTERN = "{step}_0.png"
ALIGNMENT = "rgb[t] and depth[t] are the observation before action[t]"


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _image_info(path: Path, modality: str) -> Dict:
    with Image.open(path) as image:
        array = np.asarray(image)
        mode, size, image_format = image.mode, list(image.size), image.format
    if image_format != "PNG":
        raise ValueError(f"expected PNG, got {image_format}")
    if modality == "rgb":
        if mode != "RGB" or array.shape != (size[1], size[0], 3) or array.dtype != np.uint8:
            raise ValueError(f"expected 8-bit RGB, got mode={mode} shape={array.shape} dtype={array.dtype}")
        return {"size": size, "mode": mode, "dtype": str(array.dtype)}
    if mode not in ("I;16", "I;16B", "I") or array.ndim != 2:
        raise ValueError(f"expected 16-bit grayscale depth, got mode={mode} shape={array.shape} dtype={array.dtype}")
    if not np.isfinite(array).all() or array.min() < 0 or array.max() > 65535:
        raise ValueError("depth values must be finite uint16-range integers")
    return {
        "size": size,
        "mode": mode,
        "dtype": str(array.dtype),
        "raw_range": [int(array.min()), int(array.max())],
        "valid_fraction": float(np.count_nonzero(array) / array.size),
    }


def audit_episode(episode_dir: Path, official_gt: Optional[Dict] = None) -> Dict:
    """Inspect one completed episode without changing any source file."""
    episode_dir = Path(episode_dir)
    errors: List[Dict[str, str]] = []
    stats: Dict = {"episode": str(episode_dir)}

    def error(category: str, detail: str) -> None:
        errors.append({"category": category, "detail": detail})

    required = [
        episode_dir / "READY",
        episode_dir / "metadata.json",
        episode_dir / "action" / "actions.json",
        episode_dir / "instruction" / "instruction.json",
        episode_dir / "instruction" / "tokens.npy",
        episode_dir / "trajectories" / "trajectory.json",
        episode_dir / "rgb",
        episode_dir / "depth",
    ]
    missing = [str(path.relative_to(episode_dir)) for path in required if not path.exists()]
    if missing:
        return {
            "ok": False,
            "episode": str(episode_dir),
            "errors": [{"category": "missing_files", "detail": ", ".join(missing)}],
            "stats": stats,
        }
    try:
        metadata = _load_json(episode_dir / "metadata.json")
        action_record = _load_json(episode_dir / "action" / "actions.json")
        instruction = _load_json(episode_dir / "instruction" / "instruction.json")
        trajectory = _load_json(episode_dir / "trajectories" / "trajectory.json")
    except Exception as exc:
        return {
            "ok": False,
            "episode": str(episode_dir),
            "errors": [{"category": "invalid_json", "detail": str(exc)}],
            "stats": stats,
        }

    try:
        steps = int(metadata["steps"])
        action_ids = np.asarray(action_record["actions"], dtype=np.int64)
    except Exception as exc:
        return {
            "ok": False,
            "episode": str(episode_dir),
            "errors": [{"category": "schema", "detail": f"missing or invalid steps/actions: {exc}"}],
            "stats": stats,
        }
    episode_id = str(metadata.get("episode_id", ""))
    scene_id = str(metadata.get("scene_id", ""))
    stats.update({"episode_id": episode_id, "scene_id": scene_id, "steps": steps})

    if episode_id != episode_dir.name:
        error("identity", "metadata episode_id does not match directory name")
    if str(metadata.get("scene", "")) != episode_dir.parent.name:
        error("identity", "metadata scene does not match directory name")
    if metadata.get("split") != "train":
        error("split", f"expected train, got {metadata.get('split')!r}")

    names = action_record.get("action_names")
    if names != EXPECTED_ACTION_NAMES:
        error("action_mapping", f"expected {EXPECTED_ACTION_NAMES}, got {names}")
    if action_ids.ndim != 1 or len(action_ids) != steps:
        error("action_length", f"metadata steps={steps}, action shape={action_ids.shape}")
    elif not np.isin(action_ids, np.arange(4)).all():
        error("action_range", "action id outside the declared four-action mapping")
    else:
        stops = np.flatnonzero(action_ids == 0).tolist()
        stats["action_counts"] = {
            name: int(np.count_nonzero(action_ids == index))
            for index, name in enumerate(EXPECTED_ACTION_NAMES)
        }
        stats["stop_indices"] = stops
        if stops != [steps - 1]:
            error("terminal_stop", f"expected one final STOP at {steps - 1}, got {stops}")

    if metadata.get("alignment") != ALIGNMENT:
        error("alignment", "observation/action alignment declaration is missing or unknown")
    if metadata.get("frame_pattern") != FRAME_PATTERN:
        error("alignment", "expected frame pattern {step}_0.png")
    scale = metadata.get("depth_scale")
    if not isinstance(scale, (int, float)) or scale <= 0 or metadata.get("depth_unit") != "meter":
        error("depth_scale", "depth_scale must be positive and depth_unit must be meter")

    if instruction.get("episode_id") != episode_id or instruction.get("scene_id") != scene_id:
        error("instruction_identity", "instruction episode_id/scene_id differs from metadata")
    text = instruction.get("instruction_text")
    if not isinstance(text, str) or not text.strip():
        error("instruction", "instruction_text is missing or empty")
    tokens = instruction.get("instruction_tokens")
    if not isinstance(tokens, list) or not tokens or any(type(item) is not int for item in tokens):
        error("instruction_tokens", "instruction_tokens must be a non-empty integer list")
    else:
        try:
            token_array = np.load(episode_dir / "instruction" / "tokens.npy", allow_pickle=False)
            if token_array.ndim != 1 or not np.array_equal(token_array, np.asarray(tokens)):
                error("instruction_tokens", "tokens.npy differs from instruction_tokens or is not one-dimensional")
            zero_indices = np.flatnonzero(token_array == 0)
            if len(zero_indices) and np.any(token_array[zero_indices[0]:] != 0):
                error("instruction_tokens", "padding token 0 occurs before a later non-padding token")
            stats["token_count"] = int(len(token_array))
            stats["token_nonpad"] = int(np.count_nonzero(token_array))
        except Exception as exc:
            error("instruction_tokens", f"could not read tokens.npy: {exc}")
    stats["subinstruction_fields"] = sorted(
        key for key in instruction if "sub" in key.lower() or "fsa" in key.lower()
    )

    for key in ("positions", "locations", "reference_path"):
        try:
            array = np.asarray(trajectory.get(key, []), dtype=np.float64)
            if array.ndim != 2 or (array.size and array.shape[1] != 3) or not np.isfinite(array).all():
                raise ValueError(f"expected finite [N,3], got {array.shape}")
            stats[f"trajectory_{key}_length"] = int(len(array))
        except Exception as exc:
            error("trajectory", f"{key}: {exc}")
    if len(trajectory.get("positions", [])) != steps:
        error("trajectory_alignment", "positions must contain the pre-action pose for every action")

    expected = {FRAME_PATTERN.format(step=index) for index in range(steps)}
    modality_info = {}
    for modality in ("rgb", "depth"):
        directory = episode_dir / modality
        files = {path.name for path in directory.iterdir() if path.is_file()}
        if files != expected:
            error(
                "frame_sync",
                f"{modality}: missing={sorted(expected - files)[:8]}, extra={sorted(files - expected)[:8]}",
            )
        summaries = []
        for name in sorted(files & expected, key=lambda item: int(item.split("_", 1)[0])):
            try:
                summaries.append(_image_info(directory / name, modality))
            except Exception as exc:
                error(f"{modality}_image", f"{name}: {exc}")
        if summaries:
            declared_size = metadata.get(f"{modality}_size")
            if any(item["size"] != declared_size for item in summaries):
                error(f"{modality}_size", f"images differ from declared size {declared_size}")
            modality_info[modality] = {
                "frame_count": len(files),
                "size": summaries[0]["size"],
                "mode": summaries[0]["mode"],
                "dtype": summaries[0]["dtype"],
            }
            if modality == "depth" and isinstance(scale, (int, float)) and scale > 0:
                raw_min = min(item["raw_range"][0] for item in summaries)
                raw_max = max(item["raw_range"][1] for item in summaries)
                modality_info[modality].update({
                    "raw_range": [raw_min, raw_max],
                    "meter_range": [raw_min / float(scale), raw_max / float(scale)],
                    "valid_fraction_mean": float(np.mean([item["valid_fraction"] for item in summaries])),
                })
                if raw_max == 0:
                    error("depth_empty", "all audited depth pixels are zero")
    if set(modality_info) == {"rgb", "depth"}:
        stats["modalities"] = modality_info

    if official_gt is not None:
        gt_record = official_gt.get(episode_id)
        if gt_record is None:
            error("official_gt", f"episode {episode_id} is absent from official train_gt")
        elif list(map(int, gt_record.get("actions", []))) != action_ids.tolist():
            error("official_gt", "stored actions differ from official train_gt actions")
        else:
            stats["official_gt_actions_match"] = True
    return {"ok": not errors, "episode": str(episode_dir), "errors": errors, "stats": stats}


def _load_official_gt(dataset_root: Path) -> Optional[Dict]:
    path = dataset_root / "train" / "train_gt.json.gz"
    if not path.is_file():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _episode_dirs(data_root: Path) -> List[Path]:
    episodes = []
    for scene in sorted(path for path in data_root.iterdir() if path.is_dir()):
        episodes.extend(sorted(path for path in scene.iterdir() if path.is_dir()))
    return episodes


def _select_episodes(episodes: List[Path], limit: int) -> List[Path]:
    if limit == 0 or len(episodes) <= limit:
        return episodes
    by_scene: Dict[str, List[Path]] = {}
    for path in episodes:
        by_scene.setdefault(path.parent.name, []).append(path)
    scenes = sorted(by_scene)
    if limit <= len(scenes):
        indices = np.linspace(0, len(scenes) - 1, limit, dtype=int)
        return [by_scene[scenes[index]][0] for index in indices]
    chosen = [by_scene[scene][0] for scene in scenes]
    selected = set(chosen)
    for path in episodes:
        if len(chosen) >= limit:
            break
        if path not in selected:
            chosen.append(path)
            selected.add(path)
    return chosen[:limit]


def audit_dataset(data_root: Path, max_episodes: int = 20, dataset_root: Optional[Path] = None) -> Dict:
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"data root does not exist: {data_root}")
    episodes = _episode_dirs(data_root)
    selected = _select_episodes(episodes, int(max_episodes))
    gt = _load_official_gt(Path(dataset_root)) if dataset_root is not None else None
    results = [audit_episode(path, gt) for path in selected]
    return {
        "data_root": str(data_root.resolve()),
        "dataset_root": str(Path(dataset_root).resolve()) if dataset_root is not None else None,
        "official_train_gt_available": gt is not None,
        "scene_count": len({path.parent for path in episodes}),
        "episode_directory_count": len(episodes),
        "ready_episode_count": sum((path / "READY").is_file() for path in episodes),
        "max_episodes": int(max_episodes),
        "audited_episode_count": len(results),
        "passed": sum(result["ok"] for result in results),
        "failed": sum(not result["ok"] for result in results),
        "action_mapping": EXPECTED_ACTION_NAMES,
        "results": results,
    }


def write_audit_report(report: Dict, output: Path) -> Path:
    output = Path(output)
    audit_root = Path("storage/brain_vln/audits").resolve()
    resolved = output.resolve()
    if audit_root != resolved and audit_root not in resolved.parents:
        raise ValueError(f"audit report must be written under {audit_root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output
