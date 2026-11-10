"""P01 contracts: alignment, windows, padding, reset, and action names."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from spikingnav.brain_vln.data.collate import collate_fn
from spikingnav.brain_vln.data.contract import (
    ACTION_NAMES,
    ACTION_PAD,
    BATCH_KEYS,
    FORBIDDEN_FIELDS,
    DataContractError,
    assign_scenes,
)
from spikingnav.brain_vln.data.dataset import BrainVLNDataset, scene_names, select_spotcheck_episodes
from spikingnav.brain_vln.data.episode import load_episode_record, read_header
from spikingnav.brain_vln.data.sampler import plan_windows
from spikingnav.brain_vln.data.transforms import SyncRGBDTransform

REPO = Path(__file__).resolve().parents[2]
REAL_ROOT = REPO / "storage" / "collect_data" / "train"


def make_episode(
    root: Path,
    scene: str = "scene_a",
    episode_id: str = "7",
    steps: int = 4,
    size: int = 4,
    actions=None,
    action_names=None,
    depth_scale: float = 1000.0,
    depth_unit: str = "meter",
    alignment: str = "rgb[t] and depth[t] are the observation before action[t]",
    tokens=None,
    extra_instruction=None,
    token_file=None,
    extra_rgb=None,
    skip_depth_step=None,
) -> Path:
    episode = root / scene / episode_id
    for name in ("rgb", "depth", "action", "instruction", "trajectories"):
        (episode / name).mkdir(parents=True, exist_ok=True)
    (episode / "READY").write_text("ok\n", encoding="utf-8")
    metadata = {
        "episode_id": episode_id,
        "scene_id": f"mp3d/{scene}/{scene}.glb",
        "scene": scene,
        "split": "train",
        "goal": [1.0, 2.0, 3.0],
        "geodesic_distance": 9.0,
        "start_position": [0.0, 0.0, 0.0],
        "start_rotation": [0.0, 1.0, 0.0, 0.0],
        "steps": steps,
        "rgb_size": [size, size],
        "depth_size": [size, size],
        "depth_scale": depth_scale,
        "depth_unit": depth_unit,
        "frame_pattern": "{step}_0.png",
        "alignment": alignment,
    }
    (episode / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    if actions is None:
        actions = [1] * (steps - 1) + [0]
    record = {
        "action_names": list(ACTION_NAMES) if action_names is None else list(action_names),
        "actions": list(actions),
    }
    (episode / "action" / "actions.json").write_text(json.dumps(record), encoding="utf-8")
    token_values = [4, 5, 0] if tokens is None else list(tokens)
    instruction = {
        "episode_id": episode_id,
        "scene_id": metadata["scene_id"],
        "instruction_text": "Walk forward and stop.",
        "instruction_tokens": token_values,
    }
    if extra_instruction:
        instruction.update(extra_instruction)
    (episode / "instruction" / "instruction.json").write_text(json.dumps(instruction), encoding="utf-8")
    np.save(episode / "instruction" / "tokens.npy", np.asarray(token_file or token_values, dtype=np.int32))
    trajectory = {
        "positions": [[0.0, 0.0, float(step)] for step in range(steps)],
        "locations": [],
        "reference_path": [[9.0, 9.0, 9.0]],
    }
    (episode / "trajectories" / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")
    for step in range(steps):
        rgb = np.zeros((size, size, 3), dtype=np.uint8)
        rgb[..., 0] = step
        rgb[0, 0, 1] = 7
        Image.fromarray(rgb, "RGB").save(episode / "rgb" / f"{step}_0.png")
        raw = 0 if step % 7 == 0 else 1000 + step
        Image.fromarray(np.full((size, size), raw, dtype=np.uint16)).save(episode / "depth" / f"{step}_0.png")
    if extra_rgb is not None:
        Image.fromarray(np.zeros((size, size, 3), dtype=np.uint8), "RGB").save(episode / "rgb" / extra_rgb)
    if skip_depth_step is not None:
        (episode / "depth" / f"{skip_depth_step}_0.png").unlink()
    return episode


@pytest.mark.parametrize("steps", [1, 16, 32, 63, 64, 65, 80, 100, 113, 200])
def test_windows_supervise_each_step_once(steps):
    windows = plan_windows(steps)
    supervised = []
    for spec in windows:
        assert spec.start >= 0
        assert spec.burn_in_steps == (0 if spec.start == 0 else 16)
        supervised.extend(range(spec.start + spec.burn_in_steps, spec.start + spec.valid_length))
    assert supervised == list(range(steps))


def test_short_episode_is_a_single_reset_window():
    assert plan_windows(10)[0].start == 0
    assert plan_windows(10)[0].valid_length == 10
    assert plan_windows(10)[0].burn_in_steps == 0


def test_burn_in_must_leave_room_for_a_supervised_span():
    with pytest.raises(DataContractError):
        plan_windows(80, window_size=64, burn_in=40)


def test_frames_follow_step_index_not_lexical_names(tmp_path):
    episode = make_episode(tmp_path, steps=11)
    record = load_episode_record(episode)
    lexical = sorted(path.name for path in (episode / "rgb").iterdir())
    assert lexical[2] != "2_0.png"
    assert record.rgb.shape[0] == 11
    assert torch.allclose(record.rgb[2, 0], torch.full((4, 4), 2 / 255))
    assert torch.allclose(record.rgb[10, 0], torch.full((4, 4), 10 / 255))
    assert int(record.actions[-1]) == 0
    assert torch.count_nonzero(record.actions == 0) == 1


def test_depth_uses_declared_scale_and_keeps_zero_invalid(tmp_path):
    episode = make_episode(tmp_path, steps=8, depth_scale=500.0)
    record = load_episode_record(episode)
    assert torch.equal(record.depth_valid[0], torch.zeros(1, 4, 4, dtype=torch.bool))
    assert torch.equal(record.depth[0], torch.zeros(1, 4, 4))
    assert torch.allclose(record.depth[1], torch.full((1, 4, 4), 1001 / 500))
    assert record.subinstruction_spans is None
    assert not any(hasattr(record, name) for name in FORBIDDEN_FIELDS)


def test_action_table_must_be_the_four_vln_names(tmp_path):
    reordered = make_episode(
        tmp_path / "reordered",
        action_names=["STOP", "TURN_LEFT", "MOVE_FORWARD", "TURN_RIGHT"],
    )
    objectnav = make_episode(
        tmp_path / "objectnav",
        action_names=["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "LOOK_UP", "LOOK_DOWN"],
    )
    outside = make_episode(tmp_path / "outside", steps=3, actions=[1, 4, 0])
    early_stop = make_episode(tmp_path / "early", steps=3, actions=[0, 1, 0])
    for episode in (reordered, objectnav, outside, early_stop):
        with pytest.raises(DataContractError):
            read_header(episode)


def test_alignment_rejects_missing_or_extra_frames(tmp_path):
    missing = make_episode(tmp_path / "missing", skip_depth_step=1)
    extra = make_episode(tmp_path / "extra", extra_rgb="2_1.png")
    shifted = make_episode(
        tmp_path / "shifted",
        alignment="action happens before the image",
    )
    for episode in (missing, extra, shifted):
        with pytest.raises(DataContractError):
            read_header(episode)


def test_instruction_padding_and_absent_subinstructions(tmp_path):
    interior = make_episode(tmp_path / "pad", tokens=[1, 0, 2])
    labeled = make_episode(tmp_path / "sub", extra_instruction={"subinstructions": [0, 1]})
    mismatch = make_episode(tmp_path / "npy", tokens=[4, 5, 0], token_file=[4, 6, 0])
    for episode in (interior, labeled, mismatch):
        with pytest.raises(DataContractError):
            read_header(episode)


def test_synchronized_resize_does_not_bleed_invalid_depth():
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[0, 1] = (1, 2, 3)
    depth = np.array([[0, 1000], [0, 0]], dtype=np.int64)
    native_rgb, native_depth = SyncRGBDTransform()(rgb, depth)
    assert native_rgb[0, 1, 0] == 1
    assert native_depth[0, 1] == 1000
    with pytest.raises(DataContractError):
        SyncRGBDTransform()(rgb, np.zeros((3, 2), dtype=np.int64))
    _, resized = SyncRGBDTransform(size=(4, 4))(rgb, depth)
    assert resized.shape == (4, 4)
    assert resized[0, 0] == 0
    assert resized[0, 2] == 1000


def test_short_window_pads_without_turning_pad_into_stop(tmp_path):
    episode = make_episode(tmp_path, steps=10)
    dataset = _window_dataset(tmp_path, holdout_scenes=1)
    record = dataset[0]
    assert record.rgb.shape[0] == 64
    assert int(record.valid_mask.sum()) == 10
    assert int(record.loss_mask.sum()) == 10
    assert not bool(record.burn_in_mask.any())
    assert int(record.actions[9]) == 0
    assert int(record.actions[10]) == ACTION_PAD
    assert bool(record.reset_mask[0]) and not bool(record.reset_mask[1:].any())
    assert torch.equal(record.episode_mask, record.valid_mask)
    assert record.episode_id == episode.name


def test_continuation_window_resets_and_burns_in(tmp_path):
    make_episode(tmp_path, scene="scene_a", episode_id="long", steps=100)
    make_episode(tmp_path, scene="scene_b", episode_id="other", steps=2)
    dataset = BrainVLNDataset(tmp_path, split="train", mode="window", holdout_count=1)
    assert len(dataset) == 2
    first, second = dataset[0], dataset[1]
    assert first.window_start == 0 and first.burn_in_steps == 0
    assert second.window_start == 48 and second.burn_in_steps == 16
    assert torch.allclose(second.rgb[0, 0], torch.full((4, 4), 48 / 255))
    assert not bool(second.loss_mask[:16].any())
    assert bool(second.loss_mask[16]) and not bool(second.loss_mask[52])
    assert int(torch.count_nonzero(second.loss_mask)) == 100 - 64
    batch = collate_fn([first, second])
    assert batch["rgb"].shape[0] == 64 and batch["rgb"].shape[1] == 2
    assert torch.equal(batch["reset_mask"][0], torch.ones(2, dtype=torch.bool))
    assert not bool(batch["reset_mask"][1:].any())
    assert batch["episode_id"] == ["long", "long"]
    assert tuple(batch) == BATCH_KEYS


def test_episode_mode_detaches_every_32_steps_inside_one_episode(tmp_path):
    make_episode(tmp_path, scene="scene_a", episode_id="full", steps=70)
    make_episode(tmp_path, scene="scene_b", episode_id="pad", steps=3)
    dataset = BrainVLNDataset(tmp_path, split="all", mode="episode", holdout_count=1)
    full = next(record for record in (dataset[index] for index in range(len(dataset))) if record.episode_steps == 70)
    short = next(record for record in (dataset[index] for index in range(len(dataset))) if record.episode_steps == 3)
    assert full.detach_mask[32] and full.detach_mask[64] and not full.detach_mask[0]
    batch = collate_fn([short, full])
    assert batch["valid_mask"].shape == (70, 2)
    assert int(batch["valid_mask"][:, 0].sum()) == 3
    assert int(batch["actions"][3, 0]) == ACTION_PAD
    assert bool(batch["reset_mask"][0, 0]) and bool(batch["reset_mask"][0, 1])
    assert not bool(batch["reset_mask"][1:, :].any())
    assert torch.equal(batch["episode_mask"], batch["valid_mask"])
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    loaded = next(iter(loader))
    assert loaded["rgb"].shape[1] == 2


def test_scene_holdout_is_the_sorted_tail():
    names = ["s2", "s10", "s1"]
    assignment = assign_scenes(names, holdout_count=1)
    assert assignment["train"] == ("s1", "s10")
    assert assignment["holdout"] == ("s2",)


def test_collate_rejects_mixed_spatial_sizes(tmp_path):
    left = load_episode_record(make_episode(tmp_path, scene="scene_a", episode_id="small", size=4, steps=2))
    right = load_episode_record(
        make_episode(tmp_path, scene="scene_b", episode_id="large", size=6, steps=2)
    )
    with pytest.raises(DataContractError):
        collate_fn([left, right])


def _window_dataset(root: Path, holdout_scenes: int) -> BrainVLNDataset:
    make_episode(root, scene="scene_z", episode_id="hold", steps=2)
    return BrainVLNDataset(root, split="train", mode="window", holdout_count=holdout_scenes)


@pytest.mark.skipif(not REAL_ROOT.is_dir(), reason="collected train episodes are absent")
def test_six_real_episodes_match_the_contract():
    chosen = select_spotcheck_episodes(REAL_ROOT, count=6)
    assert len(chosen) == 6
    records = [load_episode_record(path) for path in chosen]
    for path, record in zip(chosen, records):
        header = read_header(path)
        assert header.depth_scale == 1000.0
        assert header.steps == record.episode_steps == record.rgb.shape[0]
        assert record.rgb.shape[1:] == (3, 224, 224)
        assert record.depth.shape[1:] == (1, 224, 224)
        assert int(record.actions[-1]) == 0
        assert int(torch.count_nonzero(record.actions == 0)) == 1
        assert int(record.token_mask.sum()) == int(torch.count_nonzero(record.tokens))
        assert torch.isfinite(record.depth).all()
        assert torch.equal(record.depth[~record.depth_valid], torch.zeros_like(record.depth[~record.depth_valid]))
        assert record.subinstruction_spans is None
        with Image.open(path / "rgb" / "2_0.png") as image:
            expected = torch.tensor(np.transpose(np.array(image), (2, 0, 1)), dtype=torch.float32) / 255.0
        assert torch.equal(record.rgb[2], expected)
        lexical = sorted(item.name for item in (path / "rgb").iterdir() if item.is_file())
        if lexical[2] != "2_0.png":
            with Image.open(path / "rgb" / lexical[2]) as image:
                other = torch.tensor(np.transpose(np.array(image), (2, 0, 1)), dtype=torch.float32) / 255.0
            assert not torch.equal(record.rgb[2], other)
    batch = collate_fn(records[:2])
    assert tuple(batch) == BATCH_KEYS
    assert torch.equal(batch["reset_mask"][0], torch.ones(2, dtype=torch.bool))
    assert not bool(batch["reset_mask"][1:].any())
    names = scene_names(REAL_ROOT)
    assignment = assign_scenes(names)
    assert len(names) == 61
    assert assignment["holdout"] == tuple(names[-6:])
    assert set(assignment["train"]).isdisjoint(assignment["holdout"])
