"""Split a time-major batch across GPUs.

``nn.DataParallel`` cuts dimension 0. This model stores time there and the
batch on dimension 1, and the frozen CLIP encoder reads features from a
forward hook. Replicas therefore get their own hooks and a real batch slice.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate


def _accumulate(grad: torch.Tensor, dest: torch.nn.Parameter) -> torch.Tensor:
    moved = grad.detach().to(device=dest.device, dtype=dest.dtype)
    if dest.grad is None:
        dest.grad = moved.clone()
    else:
        dest.grad.add_(moved)
    return grad


def _privatize(replica: nn.Module, primary: nn.Module) -> None:
    """Give every replica its own aligned tensors.

    ``replicate`` packs weights with a coalesced broadcast, which cuDNN rejects
    as a misaligned address, and it shares neuron state across copies.
    """
    for source, copy in zip(primary.modules(), replica.modules()):
        for key, value in list(copy.__dict__.items()):
            if torch.is_tensor(value):
                copy.__dict__[key] = value.detach().contiguous().clone()
        for key, value in list(copy._buffers.items()):
            if torch.is_tensor(value):
                copy._buffers[key] = value.detach().contiguous().clone()
        for key, param in source._parameters.items():
            if param is None or not torch.is_tensor(getattr(copy, key, None)):
                continue
            cloned = getattr(copy, key).detach().contiguous().clone()
            if param.requires_grad:
                cloned.requires_grad_(True)
                cloned.register_hook(lambda grad, dest=param: _accumulate(grad, dest))
            setattr(copy, key, cloned)


def _rebind_clip_hooks(root: nn.Module) -> None:
    """Point each replica's CLIP hooks at that replica, not the original module."""
    for module in root.modules():
        clip = getattr(module, "clip_encoder", None)
        if clip is None or not hasattr(clip, "model"):
            continue
        pairs = (
            (clip.model.visual.transformer, clip._vit_hook),
            (clip.model.transformer, clip._text_hook),
        )
        for transformer, hook in pairs:
            transformer._forward_hooks = OrderedDict()
            transformer.register_forward_hook(hook)


def _chunks(count: int, parts: int) -> list[int]:
    base, extra = divmod(count, parts)
    return [base + (1 if index < extra else 0) for index in range(parts)]


def _slice_tensor(value, batch_dim: int, start: int, stop: int, device: torch.device):
    if value is None or not torch.is_tensor(value):
        return value
    sliced = value.narrow(batch_dim, start, stop - start).contiguous()
    return sliced.to(device, non_blocking=True)


def _slice_text(texts, start: int, stop: int):
    if texts is None:
        return None
    return list(texts)[start:stop]


def _cat_aux(parts: list[dict], device: torch.device) -> dict:
    keys = parts[0].keys()
    merged = {}
    for key in keys:
        values = [part[key] for part in parts]
        if values[0].ndim == 0:
            merged[key] = torch.stack([value.to(device) for value in values]).mean()
        else:
            batch_dim = 1 if values[0].ndim >= 2 else 0
            merged[key] = torch.cat([value.to(device) for value in values], dim=batch_dim)
    return merged


class BatchParallel(nn.Module):
    """Run one microbatch on several GPUs and concatenate along the batch axis."""

    def __init__(self, module: nn.Module, device_ids: list[int]) -> None:
        super().__init__()
        if len(device_ids) < 2:
            raise ValueError("BatchParallel needs at least two devices")
        self.module = module
        self.device_ids = [int(device) for device in device_ids]

    def forward(
        self,
        rgb,
        depth,
        depth_valid,
        tokens,
        token_mask,
        valid_mask,
        reset_mask,
        detach_mask,
        actions=None,
        prev_action=None,
        state=None,
        fsa_state=None,
        spans=None,
        instruction_text=None,
    ):
        batch = int(rgb.shape[1])
        devices = self.device_ids[: min(len(self.device_ids), batch)]
        replicas = replicate(self.module, devices)
        for replica in replicas:
            _privatize(replica, self.module)
            _rebind_clip_hooks(replica)
        sizes = _chunks(batch, len(devices))
        calls = []
        kwargs = []
        start = 0
        for device_index, size in zip(devices, sizes):
            stop = start + size
            device = torch.device("cuda", device_index)
            calls.append(
                (
                    _slice_tensor(rgb, 1, start, stop, device),
                    _slice_tensor(depth, 1, start, stop, device),
                    _slice_tensor(depth_valid, 1, start, stop, device),
                    _slice_tensor(tokens, 0, start, stop, device),
                    _slice_tensor(token_mask, 0, start, stop, device),
                    _slice_tensor(valid_mask, 1, start, stop, device),
                    _slice_tensor(reset_mask, 1, start, stop, device),
                    _slice_tensor(detach_mask, 1, start, stop, device),
                )
            )
            kwargs.append(
                {
                    "actions": _slice_tensor(actions, 1, start, stop, device),
                    "prev_action": _slice_tensor(prev_action, 0, start, stop, device),
                    "state": None if state is None else state.select(
                        torch.arange(start, stop, device=state.membrane.device)
                    ),
                    "fsa_state": _slice_tensor(fsa_state, 1, start, stop, device),
                    "spans": _slice_tensor(spans, 0, start, stop, device),
                    "instruction_text": _slice_text(instruction_text, start, stop),
                }
            )
            start = stop
        outputs = parallel_apply(replicas, calls, kwargs)
        home = rgb.device
        logits = torch.cat([output[0].to(home) for output in outputs], dim=1)
        aux = _cat_aux([output[2] for output in outputs], home)
        return logits, outputs[0][1], aux
