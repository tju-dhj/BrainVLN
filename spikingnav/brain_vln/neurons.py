"""LIF / ALIF steps and the two surrogate gradients used by BrainVLN."""

from __future__ import annotations

import math

import torch


class AtanSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membrane: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(membrane)
        ctx.alpha = float(alpha)
        return (membrane >= 0).to(membrane.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (membrane,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad = alpha / 2.0 / (1.0 + (math.pi / 2.0 * alpha * membrane).pow(2))
        return grad_output * grad, None


class FastSigmoidSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membrane: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(membrane)
        ctx.alpha = float(alpha)
        return (membrane >= 0).to(membrane.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (membrane,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad = 1.0 / (1.0 + alpha * membrane.abs()).pow(2)
        return grad_output * grad, None


def spike_function(membrane: torch.Tensor, kind: str, alpha: float) -> torch.Tensor:
    if kind == "atan":
        return AtanSurrogate.apply(membrane, alpha)
    if kind == "fast_sigmoid":
        return FastSigmoidSurrogate.apply(membrane, alpha)
    raise ValueError(f"unknown surrogate {kind}")


def membrane_decay(dt: float, tau_m: float) -> float:
    if dt <= 0 or tau_m <= 0:
        raise ValueError("dt and tau_m must be positive")
    return math.exp(-dt / tau_m)


def lif_step(
    membrane: torch.Tensor,
    current: torch.Tensor,
    adaptation: torch.Tensor,
    *,
    leak,
    threshold: float,
    surrogate: str,
    surrogate_alpha: float,
    adaptation_decay: float,
    adaptation_increment: float,
) -> tuple:
    """One hard-reset step.

    Adaptation is applied as an added threshold and then increased by the spike
    that this step just emitted, so the next step sees the updated threshold.
    ``adaptation_increment=0`` keeps a plain LIF.
    """
    pre = leak * membrane + current
    spike = spike_function(pre - (threshold + adaptation), surrogate, surrogate_alpha)
    updated = pre * (1.0 - spike)
    next_adaptation = adaptation_decay * adaptation + adaptation_increment * spike
    return updated, spike, next_adaptation
