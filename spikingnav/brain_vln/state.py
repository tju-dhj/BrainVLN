"""Reservoir state carried across environment steps and cleared at episode borders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class ReservoirState:
    membrane: torch.Tensor
    spike: torch.Tensor
    adaptation: torch.Tensor
    synaptic_current: torch.Tensor
    delay_queue: torch.Tensor
    time: torch.Tensor
    spatial_trace: torch.Tensor
    staleness: torch.Tensor
    hidden: torch.Tensor
    # STDP eligibility traces. Empty tensors when use_stdp is off.
    pre_trace: torch.Tensor
    post_trace: torch.Tensor
    # Slow STOP population. Width 0 on v2, so older call sites can omit it.
    stop_membrane: torch.Tensor = None

    def detach(self) -> "ReservoirState":
        return ReservoirState(
            membrane=self.membrane.detach(),
            spike=self.spike.detach(),
            adaptation=self.adaptation.detach(),
            synaptic_current=self.synaptic_current.detach(),
            delay_queue=self.delay_queue.detach(),
            time=self.time.detach(),
            spatial_trace=self.spatial_trace.detach(),
            staleness=self.staleness.detach(),
            hidden=self.hidden.detach(),
            pre_trace=self.pre_trace.detach(),
            post_trace=self.post_trace.detach(),
            stop_membrane=None if self.stop_membrane is None else self.stop_membrane.detach(),
        )

    def zero_rows(self, mask: torch.Tensor) -> "ReservoirState":
        if mask.dtype != torch.bool:
            mask = mask.bool()
        if not bool(mask.any()):
            return self
        return ReservoirState(
            membrane=_zero_rows(self.membrane, mask),
            spike=_zero_rows(self.spike, mask),
            adaptation=_zero_rows(self.adaptation, mask),
            synaptic_current=_zero_rows(self.synaptic_current, mask),
            delay_queue=_zero_rows(self.delay_queue, mask),
            time=torch.where(mask, torch.zeros_like(self.time), self.time),
            spatial_trace=_zero_rows(self.spatial_trace, mask),
            staleness=torch.where(mask, torch.zeros_like(self.staleness), self.staleness),
            hidden=_zero_rows(self.hidden, mask),
            pre_trace=_zero_rows(self.pre_trace, mask),
            post_trace=_zero_rows(self.post_trace, mask),
            stop_membrane=None if self.stop_membrane is None else _zero_rows(self.stop_membrane, mask),
        )

    def select(self, index: torch.Tensor) -> "ReservoirState":
        return ReservoirState(
            membrane=self.membrane.index_select(0, index),
            spike=self.spike.index_select(0, index),
            adaptation=self.adaptation.index_select(0, index),
            synaptic_current=self.synaptic_current.index_select(0, index),
            delay_queue=self.delay_queue.index_select(0, index),
            time=self.time.index_select(0, index),
            spatial_trace=self.spatial_trace.index_select(0, index),
            staleness=self.staleness.index_select(0, index),
            hidden=self.hidden.index_select(0, index),
            pre_trace=self.pre_trace.index_select(0, index),
            post_trace=self.post_trace.index_select(0, index),
            stop_membrane=None if self.stop_membrane is None else self.stop_membrane.index_select(0, index),
        )

    @staticmethod
    def merge(parts: Sequence["ReservoirState"]) -> "ReservoirState":
        if not parts:
            raise ValueError("cannot merge an empty state list")
        return ReservoirState(
            membrane=torch.cat([part.membrane for part in parts], dim=0),
            spike=torch.cat([part.spike for part in parts], dim=0),
            adaptation=torch.cat([part.adaptation for part in parts], dim=0),
            synaptic_current=torch.cat([part.synaptic_current for part in parts], dim=0),
            delay_queue=torch.cat([part.delay_queue for part in parts], dim=0),
            time=torch.cat([part.time for part in parts], dim=0),
            spatial_trace=torch.cat([part.spatial_trace for part in parts], dim=0),
            staleness=torch.cat([part.staleness for part in parts], dim=0),
            hidden=torch.cat([part.hidden for part in parts], dim=0),
            pre_trace=torch.cat([part.pre_trace for part in parts], dim=0),
            post_trace=torch.cat([part.post_trace for part in parts], dim=0),
            stop_membrane=(
                None
                if parts[0].stop_membrane is None
                else torch.cat([part.stop_membrane for part in parts], dim=0)
            ),
        )


def _zero_rows(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    view = mask.view(mask.shape[0], *([1] * (tensor.dim() - 1)))
    return torch.where(view, torch.zeros_like(tensor), tensor)
