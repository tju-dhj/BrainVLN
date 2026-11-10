"""LIF neuron used by SSE / SPN (paper Eqs. 1-3)."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class SurrogateHeaviside(torch.autograd.Function):
    """Heaviside with arctan surrogate gradient."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = float(alpha)
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad = alpha / 2.0 / (1.0 + (math.pi / 2.0 * alpha * x).pow(2))
        return grad_output * grad, None


class LIF(nn.Module):
    """Leaky integrate-and-fire with hard reset.

    eU_t = leak * U_{t-1} + I_t
    S_t  = H(eU_t - threshold)
    U_t  = eU_t * (1 - S_t)
    """

    def __init__(
        self,
        leak: float = 0.5,
        threshold: float = 1.0,
        surrogate_alpha: float = 2.0,
    ) -> None:
        super().__init__()
        self.leak = leak
        self.threshold = threshold
        self.surrogate_alpha = surrogate_alpha
        self.membrane: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.membrane = None

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        if self.membrane is None or self.membrane.shape != current.shape:
            self.membrane = torch.zeros_like(current)
        pre = self.leak * self.membrane + current
        spike = SurrogateHeaviside.apply(pre - self.threshold, self.surrogate_alpha)
        self.membrane = pre * (1.0 - spike)
        return spike


def reset_net(module: nn.Module) -> None:
    """Reset membrane states of all LIF neurons in a module."""
    for child in module.modules():
        if isinstance(child, LIF):
            child.reset()
