import numpy as np
import torch

from spikingnav.models.sse import TargetBackbone
from spikingnav.vlnce.episodes import mesh_path, pair_demonstration, token_length
from spikingnav.vlnce.il import imitation_loss, inflection_weights
from spikingnav.vlnce.metrics import ndtw, path_length, spl
from spikingnav.vlnce.policy import SpikingVLNPolicy
from spikingnav.vlnce.store import load_rollout, save_rollout


def test_token_length_strips_padding():
    assert token_length([4, 5, 0, 0]) == 2
    assert token_length([0, 0]) == 1


def test_pair_demonstration_keeps_expert_actions():
    episode = {
        "episode_id": 7,
        "scene_id": "mp3d/scene/scene.glb",
        "start_position": [0, 0, 0],
        "start_rotation": [0, 0, 0, 1],
        "info": {"geodesic_distance": 4.5},
        "goals": [{"position": [1, 0, 0]}],
        "instruction": {"instruction_text": "go", "instruction_tokens": [2, 3, 0]},
        "reference_path": [[0, 0, 0], [1, 0, 0]],
    }
    demo = pair_demonstration(episode, {"actions": [1, 1, 0], "locations": [[0, 0, 0]]})
    assert demo["episode_id"] == "7"
    assert demo["actions"] == [1, 1, 0]
    assert demo["geodesic_distance"] == 4.5
    assert token_length(demo["tokens"]) == 2


def test_inflection_weights_follow_vlnce():
    weights = inflection_weights(torch.tensor([1, 1, 2, 2, 0]), coef=3.2)
    assert torch.equal(weights, torch.tensor([3.2, 1.0, 3.2, 1.0, 3.2]))


def test_imitation_loss_is_small_when_the_expert_is_predicted():
    logits = torch.tensor([[[8.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 8.0]]])
    expert = torch.tensor([[0, 3]])
    valid = torch.ones(2, 1)
    weights = torch.tensor([[3.2, 1.0]])
    loss = imitation_loss(logits, expert, valid, weights)
    assert float(loss) < 0.01


def test_ndtw_of_the_same_path_is_one():
    path = [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]
    assert abs(ndtw(path, path) - 1.0) < 1e-6
    assert path_length(path) == 0.25
    assert spl(True, 1.0, 2.0) == 0.5
    assert spl(False, 1.0, 2.0) == 0.0


def test_instruction_vector_passes_through_the_target_encoder():
    target = TargetBackbone(goal_dims=32, num_categories=None)
    goal = torch.randn(3, 32)
    assert torch.equal(target(goal), goal)


def test_rollout_roundtrip(tmp_path):
    record = {
        "rgb": [np.zeros((8, 8, 3), dtype=np.uint8), np.full((8, 8, 3), 255, dtype=np.uint8)],
        "depth": [np.zeros((8, 8), dtype=np.float32), np.full((8, 8), 1.5, dtype=np.float32)],
        "actions": [1, 0],
        "tokens": [4, 5, 0],
        "instruction_text": "go left",
        "positions": [[0, 0, 0], [0.25, 0, 0]],
        "episode_id": "7",
        "scene_id": "mp3d/scene/scene.glb",
        "goal": [1, 0, 0],
        "geodesic_distance": 1.5,
        "start_position": [0, 0, 0],
        "start_rotation": [0, 0, 0, 1],
        "reference_path": [[0, 0, 0]],
        "locations": [[0, 0, 0], [0.25, 0, 0]],
    }
    root = save_rollout(str(tmp_path), record, split="train")
    assert (root / "rgb" / "0_0.png").is_file()
    assert (root / "rgb" / "1_0.png").is_file()
    assert (root / "depth" / "0_0.png").is_file()
    assert (root / "action" / "actions.json").is_file()
    assert (root / "instruction" / "instruction.json").is_file()
    assert (root / "instruction" / "tokens.npy").is_file()
    assert (root / "trajectories" / "trajectory.json").is_file()
    assert (root / "READY").is_file()
    assert not (root.parent / "7.partial").exists()
    loaded = load_rollout(str(root))
    assert loaded["actions"].tolist() == [1, 0]
    assert loaded["rgb"].shape == (2, 224, 224, 3)
    assert loaded["depth"].shape == (2, 224, 224)
    assert abs(float(loaded["depth"][1, 0, 0]) - 1.5) < 1e-3
    assert loaded["episode_id"] == "7"
    assert loaded["instruction_text"] == "go left"
    assert loaded["locations"].shape == (2, 3)


def test_policy_forward_and_backward():
    policy = SpikingVLNPolicy(torch.randn(12, 6), pretrained_ann=False)
    rgb = torch.randn(2, 1, 3, 224, 224)
    tokens = torch.randint(1, 12, (1, 5))
    lengths = torch.tensor([5])
    logits, values, memory = policy(rgb, tokens, lengths)
    assert logits.shape == (2, 1, 4)
    assert values.shape == (2, 1, 1)
    imitation_loss(logits, torch.zeros(2, 1, dtype=torch.long), torch.ones(2, 1)).backward()
    assert memory.shape[0] == 1


def test_linked_training_scene_mesh():
    path = mesh_path("data/scene_datasets", "mp3d/JeFG25nYj2p/JeFG25nYj2p.glb")
    if not path.is_file():
        return
    assert path.stat().st_size > 0
