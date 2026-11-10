"""Parameter and per-step FLOPs helpers (Table II)."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    params = module.parameters() if not trainable_only else (
        p for p in module.parameters() if p.requires_grad
    )
    return sum(p.numel() for p in params)


def _conv_macs(layer: nn.Conv2d, h: int, w: int) -> int:
    out_h = (h + 2 * layer.padding[0] - layer.dilation[0] * (layer.kernel_size[0] - 1) - 1) // layer.stride[0] + 1
    out_w = (w + 2 * layer.padding[1] - layer.dilation[1] * (layer.kernel_size[1] - 1) - 1) // layer.stride[1] + 1
    macs = out_h * out_w * layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1]
    return int(macs / max(layer.groups, 1)), out_h, out_w


def estimate_step_flops(
    module: nn.Module,
    image_size: int = 224,
    time_steps: int = 1,
    spike_rate: float = 0.2,
    snn_backbone: bool = False,
) -> Dict[str, float]:
    """Rough per-step FLOPs.

    ANN conv/linear counted as 2*MACs. SNN backbone counted as
    time_steps * spike_rate * MACs (event-driven AC ops).
    """
    macs = 0
    h = w = image_size
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            layer_macs, h, w = _conv_macs(layer, h, w)
            macs += layer_macs
        elif isinstance(layer, nn.Linear):
            macs += layer.in_features * layer.out_features
            h = w = 1
        elif isinstance(layer, (nn.AdaptiveAvgPool2d, nn.MaxPool2d, nn.AvgPool2d)):
            if isinstance(layer, nn.MaxPool2d):
                stride = layer.stride if isinstance(layer.stride, int) else layer.stride[0]
                h, w = h // stride, w // stride

    if snn_backbone:
        flops = macs * time_steps * spike_rate
    else:
        flops = 2.0 * macs
    return {
        "macs": float(macs),
        "flops": float(flops),
        "flops_g": float(flops) / 1e9,
    }


def format_param_report(module: nn.Module) -> str:
    total = count_parameters(module)
    trainable = count_parameters(module, trainable_only=True)
    return f"params={total / 1e6:.2f}M (trainable={trainable / 1e6:.2f}M)"
