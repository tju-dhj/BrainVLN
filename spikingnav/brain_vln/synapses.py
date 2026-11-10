"""Sparse directed synapses. Storage and compute follow the edge count."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

from spikingnav.brain_vln.config import BrainVLNConfig

GROUP_NAMES = ("spatial", "goal", "episodic", "action")
# Engineering topology. Cross-population edges follow the confirmed routes.
# Within-population edges give Spatial a local neighborhood and give the other
# groups a recurrent path so every neuron can carry the requested out-degree.
ALLOWED_STRUCTURED = {
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 1),
    (2, 2),
    (2, 3),
    (3, 3),
}


class SynapticGraph(nn.Module):
    def __init__(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
        weight: torch.Tensor,
        delay: torch.Tensor,
        sign: torch.Tensor,
        tau_syn: torch.Tensor,
        edge_group: torch.Tensor,
        n_neurons: int,
        dt: float,
        trainable: bool,
    ) -> None:
        super().__init__()
        if pre.numel() == 0:
            raise ValueError("synaptic graph needs at least one edge")
        if torch.any(pre == post):
            raise ValueError("synaptic graph contains a self-loop")
        if int(pre.min()) < 0 or int(post.max()) >= n_neurons:
            raise ValueError("synaptic endpoint is outside the neuron range")
        if torch.any(weight <= 0):
            raise ValueError("weights are stored positive; sign carries excitation or inhibition")
        self.n_neurons = int(n_neurons)
        self.max_delay = int(delay.max())
        self.register_buffer("pre", pre.long())
        self.register_buffer("post", post.long())
        self.register_buffer("delay", delay.long())
        self.register_buffer("sign", sign.float())
        self.register_buffer("alpha", torch.exp(torch.tensor(-dt, dtype=torch.float32) / tau_syn.float()))
        self.register_buffer("edge_group", edge_group.long())
        if trainable:
            self.weight = nn.Parameter(weight.float())
        else:
            self.register_buffer("weight", weight.float())

    @property
    def n_edges(self) -> int:
        return int(self.pre.shape[0])

    def step(self, delay_queue: torch.Tensor, time: torch.Tensor, current: torch.Tensor) -> tuple:
        """Decay every edge current and add the spike that is due at this step.

        A spike written at time ``t`` is visible when ``time - delay == t``.
        Delay 3 therefore changes the postsynaptic current on step ``t+3``.
        """
        batch, edges = current.shape
        span = delay_queue.shape[1]
        emit_time = time.view(batch, 1) - self.delay.view(1, edges)
        valid = emit_time >= 0
        slot = torch.remainder(emit_time, span)
        batch_index = torch.arange(batch, device=current.device).view(batch, 1).expand(batch, edges)
        pre = self.pre.view(1, edges).expand(batch, edges)
        delayed = delay_queue[batch_index, slot, pre]
        delayed = torch.where(valid, delayed, torch.zeros_like(delayed))
        magnitude = self.weight if not isinstance(self.weight, nn.Parameter) else self.weight.clamp(min=1e-4)
        updated = self.alpha.view(1, edges) * current + self.sign.view(1, edges) * magnitude.view(1, edges) * delayed
        post_current = torch.zeros(batch, self.n_neurons, device=current.device, dtype=updated.dtype)
        post_current.scatter_add_(1, self.post.view(1, edges).expand(batch, edges), updated)
        return updated, post_current

    def statistics(self) -> Dict[str, float]:
        out_degree = torch.bincount(self.pre, minlength=self.n_neurons).float()
        in_degree = torch.bincount(self.post, minlength=self.n_neurons).float()
        cross = self.edge_group[:, 0] != self.edge_group[:, 1]
        return {
            "edges": float(self.n_edges),
            "mean_out_degree": float(out_degree.mean()),
            "mean_in_degree": float(in_degree.mean()),
            "max_out_degree": float(out_degree.max()),
            "max_in_degree": float(in_degree.max()),
            "excitatory_fraction": float((self.sign > 0).float().mean()),
            "cross_population_fraction": float(cross.float().mean()),
        }


def build_graph(config: BrainVLNConfig, topology: str) -> SynapticGraph:
    if topology not in ("random_sparse", "functionally_structured"):
        raise ValueError(f"unknown topology {topology}")
    generator = torch.Generator()
    generator.manual_seed(int(config.seed))
    pre, post, groups = (
        _random_edges(config, generator)
        if topology == "random_sparse"
        else _structured_edges(config, generator)
    )
    edges = pre.shape[0]
    sign = torch.where(
        torch.rand(edges, generator=generator) < config.excitatory_fraction,
        torch.ones(edges),
        -torch.ones(edges),
    )
    delay = torch.randint(
        config.delay_min,
        config.delay_max + 1,
        (edges,),
        generator=generator,
    )
    weight = config.weight_scale * torch.rand(edges, generator=generator).clamp(min=1e-3)
    tau = torch.full((edges,), float(config.tau_syn))
    return SynapticGraph(
        pre,
        post,
        weight,
        delay,
        sign,
        tau,
        groups,
        config.populations.total,
        config.dt,
        config.trainable_synapses,
    )


def _random_edges(config: BrainVLNConfig, generator: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = config.populations.total
    degree = config.out_degree
    candidates = torch.arange(count)
    chosen = []
    groups = []
    group_of = _group_index(config)
    for source in range(count):
        pool = candidates[candidates != source]
        posts = _take(pool, degree, generator)
        chosen.append(posts)
        groups.append(torch.stack([group_of[source].expand(degree), group_of[posts]], dim=1))
    pre = torch.arange(count).repeat_interleave(degree)
    post = torch.cat(chosen)
    return _dedup(pre, post, torch.cat(groups))


def _structured_edges(config: BrainVLNConfig, generator: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bounds = config.populations.bounds()
    ids = {name: torch.arange(start, stop) for name, (start, stop) in bounds.items()}
    local = _spatial_local(config.populations.spatial)
    degree = config.out_degree
    pre_rows = []
    post_rows = []
    group_rows = []
    routes = {
        "spatial": torch.cat([ids["spatial"], ids["goal"], ids["episodic"], ids["action"]]),
        "goal": torch.cat([ids["episodic"], ids["action"]]),
        "episodic": torch.cat([ids["goal"], ids["episodic"], ids["action"]]),
        "action": ids["action"],
    }
    for name, pool in routes.items():
        start, stop = bounds[name]
        for source in range(start, stop):
            if name == "spatial":
                neighbors = local[source]
                merged = torch.unique(torch.cat([neighbors, pool]))
                merged = merged[merged != source]
            else:
                merged = pool[pool != source]
            if merged.numel() < 1:
                raise ValueError(f"{name} neuron {source} has no legal target")
            posts = _take(merged, min(degree, int(merged.numel())), generator)
            pre_rows.append(torch.full((posts.shape[0],), source, dtype=torch.long))
            post_rows.append(posts)
            group_rows.append(
                torch.stack(
                    [
                        torch.full((posts.shape[0],), GROUP_NAMES.index(name)),
                        _group_of_tensor(posts, bounds),
                    ],
                    dim=1,
                )
            )
    return _dedup(torch.cat(pre_rows), torch.cat(post_rows), torch.cat(group_rows))


def _spatial_local(count: int) -> Dict[int, torch.Tensor]:
    side = math.ceil(math.sqrt(count))
    coords = [(index // side, index % side) for index in range(count)]
    neighbors = {}
    for index, (row, col) in enumerate(coords):
        picked = [
            other
            for other, (other_row, other_col) in enumerate(coords)
            if other != index and max(abs(row - other_row), abs(col - other_col)) <= 2
        ]
        neighbors[index] = torch.tensor(picked, dtype=torch.long)
    return neighbors


def _take(pool: torch.Tensor, degree: int, generator: torch.Generator) -> torch.Tensor:
    if pool.numel() >= degree and pool.numel() <= 512:
        choice = torch.randperm(pool.numel(), generator=generator)[:degree]
        return pool[choice]
    draws = pool[torch.randint(0, pool.numel(), (degree,), generator=generator)]
    return draws


def _dedup(pre: torch.Tensor, post: torch.Tensor, groups: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    packed = pre.long() * (int(post.max()) + 1) + post.long()
    order = torch.argsort(packed)
    packed = packed[order]
    keep = torch.ones(packed.shape[0], dtype=torch.bool)
    keep[1:] = packed[1:] != packed[:-1]
    chosen = order[keep]
    return pre[chosen], post[chosen], groups[chosen]


def _group_index(config: BrainVLNConfig) -> torch.Tensor:
    groups = torch.empty(config.populations.total, dtype=torch.long)
    for index, (start, stop) in enumerate(config.populations.bounds().values()):
        groups[start:stop] = index
    return groups


def _group_of_tensor(posts: torch.Tensor, bounds: Dict[str, Tuple[int, int]]) -> torch.Tensor:
    group = torch.zeros_like(posts)
    for index, (start, stop) in enumerate(bounds.values()):
        group = torch.where((posts >= start) & (posts < stop), torch.full_like(posts, index), group)
    return group


def write_spike(delay_queue: torch.Tensor, time: torch.Tensor, spike: torch.Tensor) -> torch.Tensor:
    batch, span, _neurons = delay_queue.shape
    slot = torch.remainder(time, span)
    updated = delay_queue.clone()
    updated[torch.arange(batch, device=spike.device), slot] = spike
    return updated
