import torch

from spikingnav.habitat_objectnav.expert import (
    dagger_beta,
    demo_action_index,
    map_follower_action,
    viewpoint_positions,
)
from spikingnav.habitat_objectnav.il_trainer import imitation_loss


class _State:
    def __init__(self, position):
        self.position = position


class _View:
    def __init__(self, position):
        self.agent_state = _State(position)


class _Goal:
    def __init__(self, views):
        self.view_points = views
        self.position = views[0].agent_state.position


class _Episode:
    def __init__(self, goals):
        self.goals = goals


def test_map_follower_action_skips_look_actions():
    names = ["stop", "move_forward", "turn_left", "turn_right", "look_up", "look_down"]
    assert map_follower_action(0, names) == 0
    assert map_follower_action(1, names) == 1
    assert map_follower_action(3, names) == 3


def test_viewpoint_positions_reads_agent_state():
    episode = _Episode([_Goal([_View([1.0, 2.0, 3.0]), _View([4.0, 5.0, 6.0])])])
    points = viewpoint_positions(episode)
    assert len(points) == 2
    assert points[0].tolist() == [1.0, 2.0, 3.0]


def test_demo_action_skips_leading_stop():
    replay = [
        {"action": "STOP"},
        {"action": "MOVE_FORWARD"},
        {"action": "LOOK_UP"},
        {"action": "STOP"},
    ]
    names = ["stop", "move_forward", "turn_left", "turn_right", "look_up", "look_down"]
    assert demo_action_index(replay, 0, names) == (1, True)
    assert demo_action_index(replay, 1, names) == (4, True)
    assert demo_action_index(replay, 2, names) == (0, True)
    assert demo_action_index(replay, 3, names)[1] is False


def test_dagger_beta_anneals():
    assert dagger_beta(0, 100, 1.0, 0.0, 0) == 1.0
    assert dagger_beta(50, 100, 1.0, 0.0, 0) == 0.5
    assert dagger_beta(100, 100, 1.0, 0.0, 0) == 0.0
    assert dagger_beta(0, 100, 1.0, 1.0, 0) == 1.0


def test_imitation_loss_ignores_invalid_steps():
    logits = torch.tensor([[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]]], requires_grad=True)
    expert = torch.tensor([[0, 0]])
    valid = torch.tensor([[1.0, 0.0]])
    loss = imitation_loss(logits, expert, valid)
    loss.backward()
    assert loss.ndim == 0
    assert logits.grad[0, 1].abs().sum() == 0
    assert logits.grad[0, 0].abs().sum() > 0
