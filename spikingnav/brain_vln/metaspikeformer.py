"""Frozen Meta-SpikeFormer RGB encoder (Spike-driven Transformer V2).

Loads the BICLab trunk from ``third_party/Spike-Driven-Transformer-V2``.
LIF nodes are forced onto the torch backend so cupy is not required.

Weight preference under ``data/pre_trained_model/MetaSpikeFormer``:
1. Official ImageNet ``metaspikformer_8_512`` checkpoint when present.
2. Otherwise the open Spike2Former ADE20K SDTv2 backbone (same meta
   architecture, embed dims ``[64, 128, 256, 360]``), which is what this
   machine can download today (Google Drive ImageNet folder is unreachable).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = (
    Path(__file__).resolve().parents[2]
    / "third_party"
    / "Spike-Driven-Transformer-V2"
    / "classification"
)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
RGB_GRID = (3, 3)
DEFAULT_TIME_STEPS = 4
# Official 55M ImageNet model.
_IMAGENET_EMBED = [128, 256, 512, 640]
# Spike2Former ADE20K SDTv2 open backbone (Meta-SpikeFormer family).
_ADE_EMBED = [64, 128, 256, 360]


def _patch_lif_torch_backend() -> None:
    import spikingjelly.clock_driven.neuron as neuron

    if getattr(neuron, "_brain_vln_torch_backend", False):
        return
    original = neuron.MultiStepLIFNode

    def wrapped(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["backend"] = "torch"
        return original(*args, **kwargs)

    neuron.MultiStepLIFNode = wrapped  # type: ignore[misc]
    neuron._brain_vln_torch_backend = True


def _import_builder():
    _patch_lif_torch_backend()
    root = str(_REPO)
    if root not in sys.path:
        sys.path.insert(0, root)
    from functools import partial
    from models import Spiking_vit_MetaFormer, metaspikformer_8_512  # type: ignore
    import torch.nn as _nn

    def build(embed_dim, **kwargs):
        if list(embed_dim) == _IMAGENET_EMBED:
            return metaspikformer_8_512(**kwargs)
        return Spiking_vit_MetaFormer(
            img_size_h=224,
            img_size_w=224,
            patch_size=16,
            embed_dim=list(embed_dim),
            num_heads=8,
            mlp_ratios=4,
            in_channels=3,
            num_classes=0,
            qkv_bias=False,
            norm_layer=partial(_nn.LayerNorm, eps=1e-6),
            depths=8,
            sr_ratios=1,
            **kwargs,
        )

    return build


def _candidate_checkpoints(root: Path) -> list[Path]:
    preferred = (
        "metaspikformer_8_512.pth",
        "checkpoint.pth",
        "checkpoint-best.pth",
        "model_best.pth.tar",
        "metaspikeformer_spike2former_ade20k_backbone.pth",
        "ADE20K-SDTv2-ckpt-best.pth",
    )
    found = []
    for name in preferred:
        path = root / name
        if path.is_file():
            found.append(path)
    found.extend(sorted(root.glob("*.pth")))
    found.extend(sorted(root.glob("*.pth.tar")))
    unique = []
    seen = set()
    for path in found:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _strip_prefix(key: str) -> str:
    name = str(key)
    changed = True
    while changed:
        changed = False
        for prefix in (
            "module.",
            "model.",
            "backbone.",
            "encoder.",
            "student.",
            "net.",
        ):
            if name.startswith(prefix):
                name = name[len(prefix) :]
                changed = True
    return name


def _extract_state_dict(payload) -> dict:
    if not isinstance(payload, dict):
        raise RuntimeError(f"unrecognized checkpoint type: {type(payload)}")
    for key in ("state_dict", "model", "model_state_dict", "student", "network"):
        value = payload.get(key)
        if isinstance(value, dict) and value:
            payload = value
            break
    tensors = {}
    for key, value in payload.items():
        if not torch.is_tensor(value):
            continue
        name = _strip_prefix(str(key))
        if name.startswith(("head", "decode_head", "auxiliary", "cls", "mask", "neck")):
            continue
        tensors[name] = value
    if not tensors:
        raise RuntimeError("checkpoint contained no tensor weights")
    return tensors


def _infer_embed_dim(state: dict) -> list[int]:
    required = (
        "downsample1_1.encode_conv.weight",
        "downsample1_2.encode_conv.weight",
        "downsample2.encode_conv.weight",
        "downsample3.encode_conv.weight",
        "downsample4.encode_conv.weight",
    )
    missing = [key for key in required if key not in state]
    if missing:
        raise RuntimeError(f"cannot infer Meta-SpikeFormer width; missing {missing[:3]}")
    c0 = int(state["downsample1_2.encode_conv.weight"].shape[0])
    c1 = int(state["downsample2.encode_conv.weight"].shape[0])
    c2 = int(state["downsample3.encode_conv.weight"].shape[0])
    c3 = int(state["downsample4.encode_conv.weight"].shape[0])
    return [c0, c1, c2, c3]


def _load_matching(model: nn.Module, state: dict) -> Tuple[int, int]:
    current = model.state_dict()
    matched = {}
    skipped = 0
    for key, value in state.items():
        if key not in current or current[key].shape != value.shape:
            skipped += 1
            continue
        matched[key] = value
    model.load_state_dict(matched, strict=False)
    if len(matched) < 1:
        raise RuntimeError(
            "no Meta-SpikeFormer weights matched the model; "
            f"checkpoint keys={list(state)[:8]} model keys={list(current)[:8]}"
        )
    return len(matched), skipped


class FrozenMetaSpikeFormerRGB(nn.Module):
    """Frozen Meta-SpikeFormer that emits a ``[B, C, 3, 3]`` RGB map."""

    def __init__(
        self,
        root: str,
        time_steps: int = DEFAULT_TIME_STEPS,
        checkpoint: Optional[str] = None,
    ) -> None:
        super().__init__()
        if time_steps < 1:
            raise ValueError("time_steps must be positive")
        weight_root = Path(root) / "MetaSpikeFormer"
        if checkpoint:
            ckpt = Path(checkpoint)
            if not ckpt.is_file():
                raise FileNotFoundError(ckpt)
        else:
            candidates = _candidate_checkpoints(weight_root)
            if not candidates:
                raise FileNotFoundError(
                    f"no Meta-SpikeFormer checkpoint under {weight_root}"
                )
            ckpt = candidates[0]
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = _extract_state_dict(payload)
        embed_dim = _infer_embed_dim(state)
        builder = _import_builder()
        self.backbone = builder(embed_dim, kd=False)
        self.backbone.T = int(time_steps)
        matched, ignored = _load_matching(self.backbone, state)
        self.checkpoint_path = str(ckpt)
        self.matched_tensors = matched
        self.ignored_tensors = ignored
        self.embed_dim = list(embed_dim)
        self.out_channels = int(embed_dim[-1])
        self.grid = RGB_GRID
        mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean, persistent=False)
        self.register_buffer("_std", std, persistent=False)
        self.freeze()

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "FrozenMetaSpikeFormerRGB":
        super().train(False)
        self.freeze()
        return self

    def preprocess(self, rgb: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(rgb.clamp(0, 1), size=(224, 224), mode="bilinear", align_corners=False)
        return (x - self._mean) / self._std

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # LIF keeps the previous batch's membrane. The last window of an
            # epoch is smaller than the microbatch, so the stored state must
            # be cleared or the next forward dies on a shape mismatch.
            from spikingjelly.clock_driven.functional import reset_net

            reset_net(self.backbone)
            x = self.preprocess(rgb)
            x = x.unsqueeze(0).repeat(self.backbone.T, 1, 1, 1, 1)
            features = self.backbone.forward_features(x)
            if features.dim() != 5:
                raise RuntimeError(f"expected 5D features, got {tuple(features.shape)}")
            pooled = features.mean(dim=0)
            if pooled.shape[1] != self.out_channels:
                raise RuntimeError(
                    f"expected {self.out_channels} channels, got {pooled.shape[1]}"
                )
            return F.adaptive_avg_pool2d(pooled, self.grid)
