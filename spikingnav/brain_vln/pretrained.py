"""Frozen MLANet encoders in front of the spiking reservoir.

RGB is CLIP ViT-B/32 and depth is the VLN-CE ResNet-50. Language is the
instruction LSTM plus the CLIP sub-instruction tower, pooled once per
instruction. Each modality stays at the frozen 256-d width. MLANet's GRUs
are not used. The reservoir receives these vectors as current after a
fixed normalization in the model.

``FrozenMetaSpikeMLANet`` keeps the same depth and language towers, but
replaces RGB with a frozen Meta-SpikeFormer map for BrainVLN_v4.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingnav.brain_vln.metaspikeformer import FrozenMetaSpikeFormerRGB

DEPTH_METERS = 10.0
DEPTH_SIZE = 256


def habitat_depth(depth: torch.Tensor, valid: torch.Tensor, max_meters: float = DEPTH_METERS) -> torch.Tensor:
    """Map meter depth to the 0-1 range used by the Habitat DD-PPO encoder."""
    meters = depth * valid.to(dtype=depth.dtype)
    return (meters / max_meters).clamp(0.0, 1.0)


class _StubConfig:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __setstate__(self, state) -> None:
        self._state = state

    def __setitem__(self, key, value) -> None:
        store = getattr(self, "_items", None)
        if store is None:
            store = {}
            self._items = store
        store[key] = value


class _CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        root = module.split(".")[0]
        if root in {"habitat", "habitat_baselines", "habitat_sim", "yacs", "omegaconf"}:
            return type(name, (_StubConfig,), {})
        return super().find_class(module, name)


class _CheckpointPickle:
    Unpickler = _CheckpointUnpickler

    @staticmethod
    def load(file, **kwargs):
        return _CheckpointUnpickler(file).load()


def _load_state(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (AttributeError, ModuleNotFoundError):
        payload = torch.load(
            path,
            map_location="cpu",
            pickle_module=_CheckpointPickle,
            weights_only=False,
        )
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise RuntimeError(f"unrecognized checkpoint: {path}")
    tensors = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        name = str(key)
        if name.startswith("module."):
            name = name[len("module.") :]
        if name.startswith("actor_critic."):
            name = name[len("actor_critic.") :]
        tensors[name] = value
    return tensors


def _freeze(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


class FrozenMLANet(nn.Module):
    """MLANet's perception stack through its language and vision vectors."""

    def __init__(self, feature_dim: int, root: str) -> None:
        super().__init__()
        checkpoint = Path(root) / "MLANet" / "mla_best_ppo.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        from spikingnav.brain_vln.mlanet.mlanet_policy import build_mlanet

        self.policy = build_mlanet(_load_state(checkpoint))
        _freeze(self.policy)
        self.output_dim = 512
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        self._table = None
        self._instruction = None
        self._instruction_mask = None
        self._sub = None
        self._sub_mask = None

    def train(self, mode: bool = True) -> "FrozenMLANet":
        super().train(mode)
        _freeze(self.policy)
        return self

    def freeze_backbones(self) -> None:
        _freeze(self.policy)

    def prepare(self, texts, device: torch.device) -> None:
        if not texts:
            raise ValueError("MLANet language needs instruction text")
        observations = self._lookup(list(texts), device)
        net = self.policy.net
        with torch.no_grad():
            instruction = net.instruction_encoder(observations).permute(0, 2, 1)
            sub = net.clip_encoder.encode_text(observations)
            sub_mask = (sub == 0).all(dim=2)
            sub = net.sub_inst_fc(sub)
        self._instruction = instruction
        self._instruction_mask = (instruction == 0).all(dim=2)
        self._sub = sub
        self._sub_mask = sub_mask

    def encode(self, rgb: torch.Tensor, depth: torch.Tensor, valid: torch.Tensor):
        if self._instruction is None:
            raise ValueError("call prepare before the MLANet frontend")
        net = self.policy.net
        observations = {
            "rgb": (rgb.clamp(0, 1) * 255.0).permute(0, 2, 3, 1),
            "depth": self._depth_hwc(depth, valid),
        }
        with torch.no_grad():
            depth_embedding = net.depth_encoder(observations).flatten(2)
            rgb_embedding, _rgb_seq = net.clip_encoder.encode_image(observations)
            vision = torch.cat((net.rgb_fc(rgb_embedding), net.depth_fc(depth_embedding)), dim=1)
            language = self._pool_language()
        return vision, language

    def encode_visual_maps(self, rgb: torch.Tensor, depth: torch.Tensor, valid: torch.Tensor):
        """CLIP 3×3×768 RGB map and ResNet 4×4×192 depth map. Language is separate."""
        net = self.policy.net
        observations = {
            "rgb": (rgb.clamp(0, 1) * 255.0).permute(0, 2, 3, 1),
            "depth": self._depth_hwc(depth, valid),
        }
        with torch.no_grad():
            depth_map = net.depth_encoder(observations)
            _cls, rgb_seq = net.clip_encoder.encode_image(observations)
        if rgb_seq.shape[1:] != (768, 9):
            raise RuntimeError(f"expected CLIP 3x3 map as [B, 768, 9], got {tuple(rgb_seq.shape)}")
        rgb_map = rgb_seq.reshape(rgb_seq.shape[0], 768, 3, 3)
        if depth_map.dim() != 4 or depth_map.shape[1:] != (192, 4, 4):
            raise RuntimeError(f"expected depth map [B, 192, 4, 4], got {tuple(depth_map.shape)}")
        return rgb_map, depth_map

    def language_from_tokens(self, instruction: torch.Tensor, sub_instruction: torch.Tensor) -> torch.Tensor:
        """Pool MLANet's instruction LSTM and CLIP sub-instructions into 512-d.

        ``instruction`` is ``[B, 100]`` R2R ids. ``sub_instruction`` is ``[B, 12, 77]``
        CLIP token rows. This does not consult the FSA text table, so EnvDrop
        instructions that are absent from that table still encode.
        """
        net = self.policy.net
        observations = {
            "instruction": instruction.long(),
            "sub_instruction": sub_instruction.int(),
        }
        with torch.no_grad():
            encoded = net.instruction_encoder(observations).permute(0, 2, 1)
            sub = net.clip_encoder.encode_text(observations)
            sub = net.sub_inst_fc(sub)
        instruction_mask = (encoded == 0).all(dim=2)
        sub_mask = (sub == 0).all(dim=2)
        return torch.cat(
            (self._masked_mean(encoded, instruction_mask), self._masked_mean(sub, sub_mask)),
            dim=-1,
        )

    def encode_maps(self, rgb: torch.Tensor, depth: torch.Tensor, valid: torch.Tensor):
        """Spatial maps before the frozen vector projections.

        RGB is the CLIP patch grid after the encoder's 3×3 pool, ``[B, 768, 3, 3]``.
        Depth is the ResNet grid with its position embedding, ``[B, 192, 4, 4]``.
        Language is the same 512-d pooled instruction used by the vector path.
        """
        if self._instruction is None:
            raise ValueError("call prepare before the MLANet frontend")
        rgb_map, depth_map = self.encode_visual_maps(rgb, depth, valid)
        return rgb_map, depth_map, self._pool_language()

    def language_bank(self):
        """Sentence vector, the 12 sub-instruction vectors, and their padding mask.

        ``_pool_language`` averages the sub-instructions. v5 keeps each one so
        the reservoir can select the current clause. The mask is true on padding.
        """
        if self._instruction is None or self._sub is None or self._sub_mask is None:
            raise ValueError("call prepare before the MLANet frontend")
        sentence = self._masked_mean(self._instruction, self._instruction_mask)
        return sentence, self._sub, self._sub_mask

    def _pool_language(self) -> torch.Tensor:
        instruction = self._masked_mean(self._instruction, self._instruction_mask)
        sub = self._masked_mean(self._sub, self._sub_mask)
        return torch.cat((instruction, sub), dim=-1)

    @staticmethod
    def _masked_mean(sequence: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        keep = (~padding).to(dtype=sequence.dtype).unsqueeze(-1)
        return (sequence * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)

    def _depth_hwc(self, depth: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        normalized = habitat_depth(depth, valid)
        if normalized.shape[-2] != DEPTH_SIZE or normalized.shape[-1] != DEPTH_SIZE:
            normalized = F.interpolate(normalized, size=(DEPTH_SIZE, DEPTH_SIZE), mode="nearest")
        return normalized.permute(0, 2, 3, 1)

    def _lookup(self, texts, device: torch.device) -> dict:
        if self._table is None:
            from spikingnav.brain_vln.mlanet.instructions import load_official_instructions, r2r_tokens

            self._table = load_official_instructions()
            self._r2r_tokens = r2r_tokens
        instructions = []
        subs = []
        missing = []
        for text in texts:
            tokens = self._table.mlanet_tokens(str(text))
            rows = self._table.sub_tokens(str(text))
            if tokens is None or rows is None:
                missing.append(str(text)[:160])
                continue
            instructions.append(torch.tensor(self._r2r_tokens(tokens, 2504), dtype=torch.long))
            subs.append(torch.tensor(rows, dtype=torch.long))
        if missing:
            raise RuntimeError(
                "instruction is absent from R2R_VLNCE_FSASub; first=" + repr(missing[:3])
            )
        return {
            "instruction": torch.stack(instructions).to(device),
            "sub_instruction": torch.stack(subs).to(device),
        }


def _drop_module(module: nn.Module, name: str) -> None:
    if hasattr(module, name):
        delattr(module, name)


def _release_unused_mlanet(policy: nn.Module) -> None:
    """Drop every MLANet piece BrainVLN_v4 does not forward.

    The checkpoint builder only constructs the whole policy. v4 keeps the depth
    ResNet, the instruction LSTM, the CLIP text tower, and ``sub_inst_fc``.
    """
    net = policy.net
    for name in (
        "rgb_fc",
        "depth_fc",
        "prev_action_embedding",
        "low_state_encoder",
        "high_state_encoder",
        "low_level_attention",
        "high_level_attention",
        "inst_post",
        "spatial_attention_rgb",
        "spatial_attention_depth",
        "visual_post",
        "final_input_compress",
        "action_decoder",
        "progress_monitor",
    ):
        _drop_module(net, name)
    clip = net.clip_encoder
    # CLIP.dtype is a property of the vision stem. Point it at the text
    # embedding when that stem has been removed, and leave intact models alone.
    clip_cls = clip.model.__class__
    if not getattr(clip_cls, "_brain_vln_text_dtype", False):
        def _dtype(self):
            visual = self._modules.get("visual")
            if visual is not None and hasattr(visual, "conv1"):
                return visual.conv1.weight.dtype
            return self.token_embedding.weight.dtype

        clip_cls.dtype = property(_dtype)
        clip_cls._brain_vln_text_dtype = True
    visual = clip.model.visual
    visual.transformer._forward_hooks.clear()
    _drop_module(clip.model, "visual")
    _drop_module(clip, "ln_rgb")
    _drop_module(clip, "rgb_downsample")
    _drop_module(policy, "critic")
    _drop_module(policy, "action_distribution")


class FrozenMetaSpikeMLANet(FrozenMLANet):
    """MLANet depth/language with a frozen Meta-SpikeFormer RGB map.

    Keeps Goal=512 language currents and the depth 4x4x192 map from MLANet.
    RGB becomes Meta-SpikeFormer ``[B, 640, 3, 3]`` instead of CLIP 768.
    CLIP's visual tower and MLANet's navigation heads are released after their
    checkpoint has been read.
    """

    def __init__(self, feature_dim: int, root: str, time_steps: int = 4) -> None:
        super().__init__(feature_dim, root)
        _release_unused_mlanet(self.policy)
        self.rgb_encoder = FrozenMetaSpikeFormerRGB(root, time_steps=time_steps)
        self.rgb_channels = int(self.rgb_encoder.out_channels)

    def encode(self, rgb: torch.Tensor, depth: torch.Tensor, valid: torch.Tensor):
        raise RuntimeError("BrainVLN_v4 uses encode_maps with the cells layout")

    def encode_maps(self, rgb: torch.Tensor, depth: torch.Tensor, valid: torch.Tensor):
        if self._instruction is None:
            raise ValueError("call prepare before the MetaSpikeFormer frontend")
        net = self.policy.net
        observations = {
            "rgb": (rgb.clamp(0, 1) * 255.0).permute(0, 2, 3, 1),
            "depth": self._depth_hwc(depth, valid),
        }
        with torch.no_grad():
            depth_map = net.depth_encoder(observations)
            language = self._pool_language()
            rgb_map = self.rgb_encoder(rgb)
        if depth_map.dim() != 4 or depth_map.shape[1:] != (192, 4, 4):
            raise RuntimeError(f"expected depth map [B, 192, 4, 4], got {tuple(depth_map.shape)}")
        if rgb_map.shape[1:] != (self.rgb_channels, 3, 3):
            raise RuntimeError(
                f"expected RGB map [B, {self.rgb_channels}, 3, 3], got {tuple(rgb_map.shape)}"
            )
        return rgb_map, depth_map, language
