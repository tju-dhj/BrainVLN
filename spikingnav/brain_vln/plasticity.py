"""STDP and learnable delays.

Trace-based STDP follows Song et al. (2000) / NeuCube (Kasabov 2014) and the
NeuCube-Py defaults (``a_pos=1e-4``, ``a_neg=1e-2``, ``tau=3``). The sparse
edge update is the form used by BrainVLN_v3 during imitation. Learnable delays
stay outside the main imitation path.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def stdp_step(
    weight: torch.Tensor,
    pre_spike: torch.Tensor,
    post_spike: torch.Tensor,
    pre_trace: torch.Tensor,
    post_trace: torch.Tensor,
    *,
    a_pos: float = 1e-4,
    a_neg: float = 1e-2,
    tau: float = 3.0,
    dt: float = 1.0,
) -> tuple:
    """Dense one-synapse STDP step used by unit tests."""
    decay = math.exp(-dt / tau)
    pre_trace = decay * pre_trace + pre_spike
    post_trace = decay * post_trace + post_spike
    update = post_spike * (a_pos * pre_trace) - pre_spike * (a_neg * post_trace)
    return weight + update, pre_trace, post_trace


def stdp_trace_step(
    spike: torch.Tensor,
    pre_trace: torch.Tensor,
    post_trace: torch.Tensor,
    *,
    tau: float = 3.0,
    dt: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exponential eligibility traces shared by every synapse leaving/entering a neuron."""
    decay = math.exp(-dt / tau)
    return decay * pre_trace + spike, decay * post_trace + spike


def stdp_update_edges(
    weight: torch.Tensor,
    pre_index: torch.Tensor,
    post_index: torch.Tensor,
    sign: torch.Tensor,
    spike: torch.Tensor,
    pre_trace: torch.Tensor,
    post_trace: torch.Tensor,
    *,
    a_pos: float = 1e-4,
    a_neg: float = 1e-2,
    excitatory_only: bool = True,
    min_weight: float = 1e-4,
) -> torch.Tensor:
    """NeuCube-style STDP on a sparse signed graph.

    Weights are stored positive; ``sign`` carries excitation (+) or inhibition
    (-). Matching NeuCube-Py, only excitatory edges receive the update. The
    batch mean of the local rule is written into ``weight``.
    """
    pre_s = spike.index_select(1, pre_index)
    post_s = spike.index_select(1, post_index)
    pre_t = pre_trace.index_select(1, pre_index)
    post_t = post_trace.index_select(1, post_index)
    # LTP when pre was recent and post fires; LTD when post was recent and pre fires.
    delta = (a_pos * pre_t * post_s - a_neg * post_t * pre_s).mean(dim=0)
    if excitatory_only:
        delta = delta * (sign > 0).to(dtype=delta.dtype)
    return (weight + delta).clamp(min=min_weight)


class LearnableDelay(nn.Module):
    """A separate delay parameter. The main synaptic graph keeps delays fixed."""

    def __init__(self, init_delay: torch.Tensor) -> None:
        super().__init__()
        self.delay = nn.Parameter(init_delay.float())

    def quantized(self) -> torch.Tensor:
        return self.delay.clamp(min=1.0).round()
