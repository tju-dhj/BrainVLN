import torch

from spikingnav.habitat_objectnav.distributed import (
    average_gradients,
    normalize_distributed,
    preemption_triggered,
    rollout_steps_global,
)


def test_global_rollout_steps_scale_with_gpus():
    assert rollout_steps_global(128, 4, 1) == 512
    assert rollout_steps_global(128, 4, 2) == 1024


def test_normalize_matches_local_std_without_process_group():
    values = torch.tensor([[1.0, 2.0], [4.0, 7.0]])
    got = normalize_distributed(values)
    expect = (values - values.mean()) / (values.std() + 1e-8)
    assert torch.allclose(got, expect)


def test_preemption_fraction():
    assert preemption_triggered(1, 2, 0.6) is False
    assert preemption_triggered(1, 2, 0.5) is True
    assert preemption_triggered(2, 4, 0.6) is False
    assert preemption_triggered(3, 4, 0.6) is True
    assert preemption_triggered(1, 1, 0.6) is False
    assert preemption_triggered(0, 4, 0.0) is False


def test_average_gradients_is_idle_without_process_group():
    layer = torch.nn.Linear(2, 1, bias=False)
    layer.weight.grad = torch.ones_like(layer.weight)
    average_gradients(layer)
    assert torch.equal(layer.weight.grad, torch.ones_like(layer.weight))
