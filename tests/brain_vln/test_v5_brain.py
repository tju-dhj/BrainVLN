"""BrainVLN_v5 keeps the v2 reservoir and adds memory, language, and STOP paths."""

import torch

from spikingnav.brain_vln.config import BrainVLNConfig, PopulationSizes, tiny_config
from spikingnav.brain_vln.model import SPATIAL_CELL_COUNT, BrainVLNModel, modality_current
from spikingnav.brain_vln.neurons import membrane_decay


def _config() -> BrainVLNConfig:
    return BrainVLNConfig(
        populations=PopulationSizes(spatial=800, goal=512, episodic=1280, action=512),
        variant="B4",
        encoder="mlanet",
        encoder_root="data/pre_trained_model",
        visual_layout="cells",
        trainable_synapses=True,
        brain="v5",
        episodic_tau=30.0,
        stop_neurons=64,
        seed=0,
    )


def test_v5_step_uses_slow_memory_and_stop_logit():
    torch.manual_seed(0)
    model = BrainVLNModel(_config())
    model.eval()
    batch = 2
    encoded = {
        "fused": torch.randn(batch, 512),
        "instruction": torch.randn(batch, 512),
        "rgb_map": torch.randn(batch, 768, 3, 3),
        "depth_map": torch.randn(batch, 192, 4, 4),
        "sentence": torch.randn(batch, 256),
        "sub_bank": torch.randn(batch, 12, 256),
        "sub_mask": torch.zeros(batch, 12, dtype=torch.bool),
        "prev_action": torch.zeros(batch, dtype=torch.long),
    }
    encoded["sub_mask"][:, 8:] = True
    state = model.initial_state(batch, torch.device("cpu"))
    valid = torch.ones(batch, dtype=torch.bool)
    reset = torch.zeros(batch, dtype=torch.bool)
    _logits, state, _info = model._step_encoded(encoded, state, valid, reset)
    logits, state, _info = model._step_encoded(encoded, state, valid, reset)
    assert logits.shape == (batch, 4)
    assert state.stop_membrane.shape == (batch, 64)
    assert state.spatial_trace.shape == (batch, SPATIAL_CELL_COUNT)
    episodic = state.membrane[:, model.bounds["episodic"][0] : model.bounds["episodic"][1]]
    assert torch.isfinite(episodic).all()
    assert episodic.abs().mean() > 0
    slow = membrane_decay(1.0, 30.0 * model.config.internal_steps)
    fast = membrane_decay(1.0, model.config.tau_m)
    assert abs(model.neuron_leak[model.bounds["episodic"][0]].item() - slow) < 1e-6
    assert abs(model.neuron_leak[model.bounds["action"][0]].item() - fast) < 1e-6
    assert model.readout.in_features == 512 + 512 + 1280
    assert model.graph.n_edges == 147774

    zero = model.initial_state(batch, torch.device("cpu"))
    opened = model._goal_current(encoded, zero)
    mean_sub = encoded["sub_bank"][:, :8].mean(dim=1)
    expected = modality_current(torch.cat((encoded["sentence"], mean_sub), dim=-1))
    assert torch.allclose(opened, expected, atol=1e-5)

    with torch.no_grad():
        same, _surprise, *_rest = model._external(encoded, state, None, None)
        start, stop = model.bounds["episodic"]
        assert same[:, start:stop].abs().sum() == 0
        changed = dict(encoded)
        changed["rgb_map"] = torch.randn_like(encoded["rgb_map"])
        moved, delta, *_rest = model._external(changed, state, None, None)
        assert float(delta.min()) > model.config.view_change_threshold
        assert moved[:, start:stop].abs().sum() > 0

    logits.sum().backward()
    assert model.sub_pointer.weight.grad.abs().sum() > 0
    assert model.stop_logit.weight.grad.abs().sum() > 0
    assert model.episodic_rest.weight.grad.abs().sum() > 0

    blank = dict(encoded)
    blank["sub_mask"] = torch.ones(batch, 12, dtype=torch.bool)
    finite, _state, _info = model._step_encoded(
        blank, model.initial_state(batch, torch.device("cpu")), valid, reset
    )
    assert torch.isfinite(finite).all()


def test_v2_does_not_grow_the_checkpoint():
    model = BrainVLNModel(tiny_config(variant="B4"))
    assert model.v5 is False
    assert "neuron_leak" not in model.state_dict()
    assert model.readout.in_features == 8
    state = model.initial_state(2, torch.device("cpu"))
    assert tuple(state.stop_membrane.shape) == (2, 0)
