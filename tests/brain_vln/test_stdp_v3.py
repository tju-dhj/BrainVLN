"""BrainVLN_v3: NeuCube-style STDP on the cells-layout B4 graph."""

from __future__ import annotations

import torch

from spikingnav.brain_vln.config import PopulationSizes, tiny_config
from spikingnav.brain_vln.model import BrainVLNModel
from spikingnav.brain_vln.plasticity import stdp_trace_step, stdp_update_edges


def test_excitatory_edges_potentiate_when_pre_then_post():
    weight = torch.tensor([0.2, 0.2])
    sign = torch.tensor([1.0, -1.0])
    pre = torch.tensor([0, 0])
    post = torch.tensor([1, 2])
    spike0 = torch.tensor([[1.0, 0.0, 0.0]])
    pre_t = torch.zeros(1, 3)
    post_t = torch.zeros(1, 3)
    pre_t, post_t = stdp_trace_step(spike0, pre_t, post_t, tau=3.0)
    spike1 = torch.tensor([[0.0, 1.0, 1.0]])
    pre_t, post_t = stdp_trace_step(spike1, pre_t, post_t, tau=3.0)
    updated = stdp_update_edges(
        weight.clone(),
        pre,
        post,
        sign,
        spike1,
        pre_t,
        post_t,
        a_pos=0.01,
        a_neg=0.012,
    )
    assert float(updated[0]) > float(weight[0])
    assert float(updated[1]) == float(weight[1])


def test_v3_wires_stdp_traces_and_respects_allow_gate():
    torch.manual_seed(0)
    config = tiny_config(
        variant="B4",
        encoder="conv",
        trainable_synapses=True,
        use_stdp=True,
        stdp_a_pos=0.2,
        stdp_a_neg=0.0,
        populations=PopulationSizes(spatial=8, goal=4, episodic=6, action=4),
    )
    model = BrainVLNModel(config)
    model.train()
    assert model.use_stdp is True
    steps, batch, size = 4, 2, 16
    rgb = torch.rand(steps, batch, 3, size, size)
    depth = torch.rand(steps, batch, 1, size, size)
    depth_valid = torch.ones(steps, batch, 1, size, size, dtype=torch.bool)
    tokens = torch.randint(1, 16, (batch, 4))
    token_mask = torch.ones(batch, 4, dtype=torch.bool)
    valid = torch.ones(steps, batch, dtype=torch.bool)
    reset = torch.zeros(steps, batch, dtype=torch.bool)
    reset[0] = True
    detach = torch.zeros(steps, batch, dtype=torch.bool)
    actions = torch.randint(0, 4, (steps, batch))

    before = model.graph.weight.detach().clone()
    model.allow_stdp = True
    _logits, state, aux = model(
        rgb,
        depth,
        depth_valid,
        tokens,
        token_mask,
        valid,
        reset,
        detach,
        actions=actions,
    )
    after_first = model.graph.weight.detach().clone()
    assert state.pre_trace.shape == (batch, config.populations.total)
    assert state.post_trace.shape == (batch, config.populations.total)
    assert float(aux["firing_rate"].sum()) > 0.0
    assert float((after_first - before).abs().max()) > 0.0

    model.allow_stdp = False
    _logits, _state, _aux = model(
        rgb,
        depth,
        depth_valid,
        tokens,
        token_mask,
        valid,
        reset,
        detach,
        actions=actions,
    )
    assert torch.equal(after_first, model.graph.weight.detach())
