import json
from pathlib import Path

import numpy as np
from PIL import Image

from spikingnav.vln.data.audit import audit_episode


def make_episode(root: Path) -> Path:
    episode = root / "scene_a" / "7"
    for name in ("rgb", "depth", "action", "instruction", "trajectories"):
        (episode / name).mkdir(parents=True, exist_ok=True)
    (episode / "READY").write_text("ok\n")
    metadata = {
        "episode_id": "7",
        "scene_id": "mp3d/scene_a/scene_a.glb",
        "scene": "scene_a",
        "split": "train",
        "steps": 2,
        "rgb_size": [8, 8],
        "depth_size": [8, 8],
        "depth_scale": 1000.0,
        "depth_unit": "meter",
        "frame_pattern": "{step}_0.png",
        "alignment": "rgb[t] and depth[t] are the observation before action[t]",
    }
    (episode / "metadata.json").write_text(json.dumps(metadata))
    actions = {"action_names": ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"], "actions": [1, 0]}
    (episode / "action" / "actions.json").write_text(json.dumps(actions))
    instruction = {
        "episode_id": "7",
        "scene_id": metadata["scene_id"],
        "instruction_text": "Walk forward and stop.",
        "instruction_tokens": [2, 3, 0],
    }
    (episode / "instruction" / "instruction.json").write_text(json.dumps(instruction))
    np.save(episode / "instruction" / "tokens.npy", np.asarray([2, 3, 0], dtype=np.int32))
    trajectory = {"positions": [[0, 0, 0], [0.25, 0, 0]], "locations": [], "reference_path": []}
    (episode / "trajectories" / "trajectory.json").write_text(json.dumps(trajectory))
    for step in range(2):
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), "RGB").save(episode / "rgb" / f"{step}_0.png")
        Image.fromarray(np.full((8, 8), 1500, dtype=np.uint16)).save(episode / "depth" / f"{step}_0.png")
    return episode


def test_audit_accepts_aligned_rgbd_and_final_stop(tmp_path):
    report = audit_episode(make_episode(tmp_path))
    assert report["ok"], report["errors"]
    assert report["stats"]["steps"] == 2
    assert report["stats"]["stop_indices"] == [1]
    assert report["stats"]["modalities"]["depth"]["meter_range"] == [1.5, 1.5]


def test_audit_rejects_missing_depth_frame(tmp_path):
    episode = make_episode(tmp_path)
    (episode / "depth" / "1_0.png").unlink()
    report = audit_episode(episode)
    assert not report["ok"]
    assert any(issue["category"] == "frame_sync" for issue in report["errors"])


def test_audit_rejects_wrong_terminal_action(tmp_path):
    episode = make_episode(tmp_path)
    actions_path = episode / "action" / "actions.json"
    action_record = json.loads(actions_path.read_text())
    action_record["actions"][-1] = 1
    actions_path.write_text(json.dumps(action_record))
    report = audit_episode(episode)
    assert not report["ok"]
    assert any(issue["category"] == "terminal_stop" for issue in report["errors"])

