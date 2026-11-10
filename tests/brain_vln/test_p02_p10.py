"""P02-P10 checks that do not start a full training run or a DAgger collection."""

import math
from pathlib import Path

import pytest
import torch

from spikingnav.brain_vln.closed_loop import ClosedLoopActor, prepare_dagger, score_trajectory
from spikingnav.brain_vln.config import BrainVLNConfig, PopulationSizes, tiny_config
from spikingnav.brain_vln.flywire import degree_preserving_rewire, flywire_status
from spikingnav.brain_vln.frontend import Frontend
from spikingnav.brain_vln.pretrained import habitat_depth
from spikingnav.brain_vln.lesion import population_mask, random_mask
from spikingnav.brain_vln.losses import masked_action_ce
from spikingnav.brain_vln.memory import EventMemory
from spikingnav.brain_vln.model import ACTION_SCALE, BrainVLNModel, modality_current
from spikingnav.brain_vln.neurons import lif_step, membrane_decay
from spikingnav.brain_vln.plasticity import LearnableDelay, stdp_step
from spikingnav.brain_vln.probe import counterfactual_probe
from spikingnav.brain_vln.state import ReservoirState
from spikingnav.brain_vln.synapses import ALLOWED_STRUCTURED, SynapticGraph, build_graph, write_spike
from spikingnav.brain_vln.train_loop import load_checkpoint, save_checkpoint, train_batch


def _batch(steps=4, batch=2, size=8, vocab=64):
    generator = torch.Generator().manual_seed(0)
    rgb = torch.rand(steps, batch, 3, size, size, generator=generator)
    depth = torch.rand(steps, batch, 1, size, size, generator=generator)
    depth_valid = torch.ones(steps, batch, 1, size, size, dtype=torch.bool)
    depth_valid[:, :, :, 0, 0] = False
    tokens = torch.randint(1, vocab, (batch, 6), generator=generator)
    token_mask = torch.ones(batch, 6, dtype=torch.bool)
    token_mask[:, -1] = False
    actions = torch.randint(0, 4, (steps, batch), generator=generator)
    valid = torch.ones(steps, batch, dtype=torch.bool)
    reset = torch.zeros(steps, batch, dtype=torch.bool)
    reset[0] = True
    detach = torch.zeros(steps, batch, dtype=torch.bool)
    detach[min(2, steps - 1)] = True
    return {
        "rgb": rgb,
        "depth": depth,
        "depth_valid": depth_valid,
        "tokens": tokens,
        "token_mask": token_mask,
        "actions": actions,
        "valid_mask": valid,
        "reset_mask": reset,
        "detach_mask": detach,
        "loss_mask": valid.clone(),
        "window_start": torch.zeros(batch, dtype=torch.long),
        "episode_steps": torch.full((batch,), steps, dtype=torch.long),
    }


def test_lif_and_alif_follow_the_locked_constants():
    leak = membrane_decay(1.0, 2.0)
    assert leak == pytest.approx(math.exp(-0.5))
    membrane, spike, adaptation = lif_step(
        torch.zeros(1),
        torch.tensor([2.0]),
        torch.zeros(1),
        leak=leak,
        threshold=1.0,
        surrogate="atan",
        surrogate_alpha=2.0,
        adaptation_decay=0.9,
        adaptation_increment=0.1,
    )
    assert float(spike) == 1.0
    assert float(membrane) == 0.0
    assert float(adaptation) == pytest.approx(0.1)
    value = torch.tensor([0.2], requires_grad=True)
    fast = lif_step(
        torch.zeros(1),
        value,
        torch.zeros(1),
        leak=leak,
        threshold=1.0,
        surrogate="fast_sigmoid",
        surrogate_alpha=2.0,
        adaptation_decay=0.9,
        adaptation_increment=0.0,
    )
    fast[1].sum().backward()
    assert torch.isfinite(value.grad)


def test_single_synapse_delay_decay_and_sign():
    graph = SynapticGraph(
        pre=torch.tensor([0, 0]),
        post=torch.tensor([1, 1]),
        weight=torch.tensor([0.5, 0.25]),
        delay=torch.tensor([3, 3]),
        sign=torch.tensor([1.0, -1.0]),
        tau_syn=torch.tensor([2.0, 2.0]),
        edge_group=torch.tensor([[0, 0], [0, 0]]),
        n_neurons=2,
        dt=1.0,
        trainable=False,
    )
    queue = torch.zeros(1, graph.max_delay + 1, 2)
    current = torch.zeros(1, 2)
    clock = torch.zeros(1, dtype=torch.long)
    for _ in range(3):
        current, post = graph.step(queue, clock, current)
        assert float(post[0, 1]) == 0.0
        queue = write_spike(queue, clock, torch.tensor([[1.0, 0.0]]))
        clock = clock + 1
    current, post = graph.step(queue, clock, current)
    assert float(post[0, 1]) == pytest.approx(0.25)
    decayed, _ = graph.step(torch.zeros_like(queue), clock + 10, current)
    assert float(decayed[0, 0]) == pytest.approx(math.exp(-0.5) * float(current[0, 0]))
    assert graph.weight.numel() == graph.n_edges
    assert graph.pre.numel() == graph.n_edges


def test_state_reset_select_and_merge():
    model = BrainVLNModel(tiny_config(variant="B2"))
    state = model.initial_state(2, torch.device("cpu"))
    state.membrane = torch.ones_like(state.membrane)
    state.delay_queue = torch.ones_like(state.delay_queue)
    cleared = state.zero_rows(torch.tensor([False, True]))
    assert float(cleared.membrane[0].sum()) > 0
    assert float(cleared.membrane[1].sum()) == 0
    assert float(cleared.delay_queue[1].sum()) == 0
    restored = ReservoirState.merge([cleared.select(torch.tensor([1])), cleared.select(torch.tensor([0]))])
    assert float(restored.membrane[0].sum()) == 0
    assert float(restored.membrane[1].sum()) > 0
    assert restored.detach().membrane.requires_grad is False


def test_degree_targets_and_structured_routes():
    for degree in (2, 4, 6):
        config = tiny_config(out_degree=degree, populations=PopulationSizes(6, 4, 5, 4))
        graph = build_graph(config, "random_sparse")
        stats = graph.statistics()
        assert stats["edges"] <= 19 * degree
        assert stats["mean_out_degree"] == pytest.approx(degree)
        assert 0.6 < stats["excitatory_fraction"] < 0.95
    structured = build_graph(tiny_config(), "functionally_structured")
    pairs = {(int(row[0]), int(row[1])) for row in structured.edge_group.tolist()}
    assert pairs <= ALLOWED_STRUCTURED
    assert (0, 1) in pairs
    assert (2, 1) in pairs


def test_frontend_mask_freeze_and_empty_subinstruction():
    config = tiny_config()
    front = Frontend(config)
    rgb = torch.rand(2, 3, 8, 8)
    depth = torch.rand(2, 1, 8, 8)
    valid = torch.ones(2, 1, 8, 8, dtype=torch.bool)
    tokens = torch.tensor([[1, 2, 3, 0], [1, 2, 4, 0]])
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=torch.bool)
    changed = tokens.clone()
    changed[:, 2] = 9
    prev = torch.full((2,), 4)
    left = front(rgb, depth, valid, tokens, mask, prev, None)
    right = front(rgb, depth, valid, changed, mask, prev, None)
    assert torch.allclose(left["instruction"], right["instruction"])
    assert torch.equal(left["subinstruction"], torch.zeros_like(left["subinstruction"]))
    front.set_frozen(rgb=True)
    loss = front(rgb, depth, valid, tokens, mask, prev, None)["fused"].sum()
    loss.backward()
    assert front.rgb.proj.weight.grad is None
    assert front.depth.proj.weight.grad is not None


def test_variants_padding_and_counterfactual():
    batch = _batch()
    batch["valid_mask"][-1] = False
    batch["loss_mask"][-1] = False
    for variant in ("B0", "B1", "B2", "B3", "B4", "B5"):
        model = BrainVLNModel(tiny_config(variant=variant))
        logits, state, aux = model(
            batch["rgb"],
            batch["depth"],
            batch["depth_valid"],
            batch["tokens"],
            batch["token_mask"],
            batch["valid_mask"],
            batch["reset_mask"],
            batch["detach_mask"],
            actions=batch["actions"],
        )
        assert logits.shape == (4, 2, 4)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(state.membrane).all()
        assert float(aux["firing_rate"].min()) >= 0
    separated = BrainVLNModel(tiny_config(variant="B4"))
    probe = counterfactual_probe(
        separated,
        batch["rgb"][:, :1],
        batch["depth"][:, :1],
        batch["depth_valid"][:, :1],
        batch["tokens"][:1],
        batch["token_mask"][:1],
        torch.tensor([[8, 9, 10, 11, 12, 0]]),
        torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.bool),
    )
    assert probe["spatial_equal"]
    assert probe["goal_changed"]
    assert probe["goal_abs_diff"] > probe["spatial_abs_diff"]


def test_event_gate_skips_closed_writes_and_respects_staleness():
    memory = EventMemory(8, mode="surprise", threshold=1e6, max_staleness=3, skip_inactive=True)
    spatial = torch.zeros(2, 8)
    previous = torch.zeros(2, 8)
    goal = torch.zeros(2, 8)
    episodic = torch.zeros(2, 8)
    closed = memory(spatial, previous, goal, episodic, spatial, torch.zeros(2), torch.zeros(2, dtype=torch.long), None, None)
    assert float(closed["gate"].sum()) == 0
    assert memory.executed_writes == 0
    forced = memory(
        spatial,
        previous,
        goal,
        episodic,
        spatial,
        torch.full((2,), 3.0),
        torch.zeros(2, dtype=torch.long),
        None,
        None,
    )
    assert float(forced["gate"].sum()) == 2
    assert memory.executed_writes == 2
    fsa = EventMemory(8, mode="task-aware", threshold=1e6, max_staleness=100)
    switched = fsa(
        spatial,
        previous,
        goal,
        episodic,
        spatial,
        torch.zeros(2),
        torch.zeros(2, dtype=torch.long),
        torch.tensor([1, 1]),
        torch.tensor([0, 1]),
    )
    assert switched["task_switch"].tolist() == [1.0, 0.0]


def test_loss_ignores_padding_and_checkpoint_roundtrip(tmp_path):
    logits = torch.zeros(2, 1, 4, requires_grad=True)
    actions = torch.tensor([[1], [-1]])
    mask = torch.tensor([[1.0], [0.0]])
    loss = masked_action_ce(logits, actions, mask)
    assert loss.ndim == 0
    model = BrainVLNModel(tiny_config(variant="B1"))
    batch = _batch(steps=2, batch=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    _state, parts = train_batch(model, batch, optimizer)
    assert "action_ce" in parts
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, optimizer, 1, parts)
    restored = BrainVLNModel(tiny_config(variant="B1"))
    payload = load_checkpoint(path, restored, optimizer)
    assert payload["step"] == 1


def test_closed_loop_actor_has_no_goal_argument():
    assert "goal" not in ClosedLoopActor.act.__code__.co_varnames
    model = BrainVLNModel(tiny_config(variant="B0"))
    actor = ClosedLoopActor(model)
    action = actor.act(
        torch.rand(3, 8, 8),
        torch.rand(1, 8, 8),
        torch.ones(1, 8, 8, dtype=torch.bool),
        torch.tensor([1, 2, 3]),
        torch.tensor([1, 1, 1], dtype=torch.bool),
    )
    assert action in (0, 1, 2, 3)
    scored = score_trajectory([[0, 0, 0], [0.25, 0, 0]], [[0, 0, 0], [0.25, 0, 0]], True, 1.0, 4.0)
    assert scored["success"] is True
    with pytest.raises(ValueError):
        prepare_dagger(Path("/tmp/not-dagger"), True)
    with pytest.raises(RuntimeError):
        prepare_dagger(Path("storage/brain_vln/dagger/pending"), False)


def test_stdp_and_fixed_mainline_delays():
    weight = torch.tensor([0.2])
    updated, pre_trace, _post_trace = stdp_step(
        weight,
        torch.tensor([1.0]),
        torch.tensor([0.0]),
        torch.zeros(1),
        torch.zeros(1),
    )
    updated, _pre_trace, _post_trace = stdp_step(
        updated,
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        pre_trace,
        torch.zeros(1),
    )
    assert float(updated) > float(weight)
    delay = LearnableDelay(torch.tensor([2.0]))
    delay.quantized().sum().backward()
    assert delay.delay.grad is not None
    model = BrainVLNModel(tiny_config(variant="B3"))
    assert model.graph.delay.requires_grad is False
    assert "weight" not in dict(model.graph.named_parameters())


def test_flywire_missing_extract_and_rewire():
    status = flywire_status(Path("storage/brain_vln/flywire"))
    assert status["status"] == "NOT_RUN"
    pre = torch.tensor([0, 0, 1])
    post = torch.tensor([2, 3, 4])
    new_pre, new_post = degree_preserving_rewire(pre, post, seed=1)
    assert torch.equal(torch.bincount(new_pre), torch.bincount(pre))
    assert torch.equal(torch.sort(new_post).values, torch.sort(post).values)


def test_population_lesion_matches_the_requested_count():
    config = tiny_config()
    goal = population_mask(config, "goal")
    random = random_mask(config, int(goal.sum()), seed=3)
    assert int(goal.sum()) == config.populations.goal
    assert int(random.sum()) == int(goal.sum())
    assert not torch.equal(goal, random)


def test_long_tiny_rollout_stays_finite():
    model = BrainVLNModel(tiny_config(variant="B5", event_mode="no-gate"))
    batch = _batch(steps=12, batch=2)
    logits, state, aux = model(
        batch["rgb"],
        batch["depth"],
        batch["depth_valid"],
        batch["tokens"],
        batch["token_mask"],
        batch["valid_mask"],
        batch["reset_mask"],
        batch["detach_mask"],
        actions=batch["actions"],
    )
    assert torch.isfinite(logits).all()
    assert torch.isfinite(state.membrane).all()
    assert torch.isfinite(aux["surprise"]).all()


def test_modality_current_normalizes_each_half():
    generator = torch.Generator().manual_seed(0)
    rgb = torch.randn(8, 256, generator=generator) * 40.0 + 12.0
    depth = torch.randn(8, 256, generator=generator) * 0.02
    current = modality_current(torch.cat((rgb, depth), dim=-1))
    rgb_std = current[:, :256].std(dim=-1, unbiased=False)
    depth_std = current[:, 256:].std(dim=-1, unbiased=False)
    assert torch.allclose(rgb_std, torch.full_like(rgb_std, 0.5), atol=1e-3)
    assert torch.allclose(depth_std, torch.full_like(depth_std, 0.5), atol=1e-2)
    assert current[:, :256].mean(dim=-1).abs().max() < 1e-4
    assert current[:, 256:].mean(dim=-1).abs().max() < 1e-4
    action = ACTION_SCALE * torch.tanh(torch.randn(8, 512, generator=generator) * 10.0)
    assert action.abs().max() <= ACTION_SCALE + 1e-6


def test_habitat_depth_uses_meters_and_invalid_pixels():
    depth = torch.tensor([[[[0.0, 10.0], [20.0, 5.0]]]])
    valid = torch.tensor([[[[False, True], [True, True]]]])
    normalized = habitat_depth(depth, valid)
    assert normalized[0, 0, 0, 0].item() == 0.0
    assert normalized[0, 0, 0, 1].item() == pytest.approx(1.0)
    assert normalized[0, 0, 1, 0].item() == pytest.approx(1.0)
    assert normalized[0, 0, 1, 1].item() == pytest.approx(0.5)
