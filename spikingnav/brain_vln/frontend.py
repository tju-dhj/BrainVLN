"""RGB, depth, and instruction encoders. Sub-instruction spans stay empty."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from spikingnav.brain_vln.config import BrainVLNConfig
from spikingnav.brain_vln.pretrained import FrozenMLANet, FrozenMetaSpikeMLANet


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, feature_dim: int) -> None:
        super().__init__()
        self.tower = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=7, stride=4, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=4, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(32, feature_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        flat = self.tower(frames).flatten(1)
        return self.proj(flat)


class LocalCrossAttention(nn.Module):
    """Two-token attention between the RGB vector and the depth vector."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(feature_dim, 1)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack((rgb, depth), dim=1)
        weights = torch.softmax(self.score(tokens).squeeze(-1), dim=1)
        return (tokens * weights.unsqueeze(-1)).sum(dim=1)


class RGBDFusion(nn.Module):
    def __init__(self, feature_dim: int, cross_attention: bool) -> None:
        super().__init__()
        self.cross = LocalCrossAttention(feature_dim) if cross_attention else None
        self.proj = nn.Linear(feature_dim * 2, feature_dim)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        if self.cross is not None:
            return self.cross(rgb, depth)
        return self.proj(torch.cat((rgb, depth), dim=-1))


class InstructionEncoder(nn.Module):
    def __init__(self, vocab_size: int, feature_dim: int, pad_index: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, feature_dim, padding_idx=pad_index)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if int(tokens.max()) >= self.vocab_size or int(tokens.min()) < 0:
            raise ValueError(
                f"token id outside 0..{self.vocab_size - 1}; refusing to wrap the vocabulary"
            )
        embedded = self.embed(tokens.long())
        weights = mask.to(embedded.dtype).unsqueeze(-1)
        return (embedded * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


class SubinstructionEncoder(nn.Module):
    """Returns an empty code unless explicit spans are supplied.

    Collected episodes do not contain FSA boundaries. This module does not
    invent them.
    """

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.proj = nn.Linear(feature_dim, feature_dim)

    def forward(
        self,
        instruction: torch.Tensor,
        spans: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = instruction.shape[0]
        present = instruction.new_zeros(batch)
        if spans is None:
            return instruction.new_zeros(batch, self.feature_dim), present
        # spans: [B, 2] inclusive-exclusive token bounds already pooled upstream.
        present = torch.ones(batch, device=instruction.device)
        return self.proj(instruction), present


class Frontend(nn.Module):
    def __init__(self, config: BrainVLNConfig) -> None:
        super().__init__()
        if config.encoder not in ("conv", "mlanet", "metaspikeformer"):
            raise ValueError("encoder must be conv, mlanet, or metaspikeformer")
        self.encoder_name = config.encoder
        self.visual_layout = config.visual_layout
        self.brain = config.brain
        self.fusion = RGBDFusion(config.feature_dim, config.cross_attention)
        self.subinstruction = SubinstructionEncoder(config.feature_dim)
        self.action_embed = nn.Embedding(5, config.feature_dim)
        self.weight_source = config.weight_source
        self._language = None
        self.pretrained = None
        if config.encoder == "conv":
            self.rgb = ConvEncoder(3, config.feature_dim)
            self.depth = ConvEncoder(1, config.feature_dim)
            self.instruction = InstructionEncoder(config.vocab_size, config.feature_dim, config.pad_index)
        elif config.encoder == "metaspikeformer":
            if config.visual_layout != "cells":
                raise ValueError("metaspikeformer encoder requires visual_layout=cells")
            self.pretrained = FrozenMetaSpikeMLANet(config.feature_dim, config.encoder_root)
        else:
            self.pretrained = FrozenMLANet(config.feature_dim, config.encoder_root)

    def train(self, mode: bool = True) -> "Frontend":
        super().train(mode)
        if self.pretrained is not None:
            self.pretrained.freeze_backbones()
        return self

    def prepare_language(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        instruction_text=None,
    ) -> None:
        if self.pretrained is None:
            self._language = None
            return
        self.pretrained.prepare(instruction_text, tokens.device)

    def set_frozen(self, rgb: bool = False, depth: bool = False, instruction: bool = False) -> None:
        if self.pretrained is not None:
            self.pretrained.freeze_backbones()
            return
        _freeze(self.rgb, rgb)
        _freeze(self.depth, depth)
        _freeze(self.instruction, instruction)

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        depth_valid: torch.Tensor,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        prev_action: torch.Tensor,
        spans: Optional[torch.Tensor] = None,
        reset: Optional[torch.Tensor] = None,
    ) -> dict:
        rgb_map = None
        depth_map = None
        if self.pretrained is None:
            visual_rgb = self.rgb(rgb)
            visual_depth = self.depth(depth * depth_valid.to(depth.dtype))
            words = self.instruction(tokens, token_mask)
            fused = self.fusion(visual_rgb, visual_depth)
        else:
            if self.visual_layout == "cells":
                rgb_map, depth_map, words = self.pretrained.encode_maps(rgb, depth, depth_valid)
                visual_rgb = words
                visual_depth = words
                fused = words
            else:
                visual_rgb, words = self.pretrained.encode(rgb, depth, depth_valid)
                visual_depth = visual_rgb
                fused = visual_rgb
        sub_code, sub_present = self.subinstruction(words, spans)
        history = self.action_embed(prev_action.long().clamp(0, 4))
        encoded = {
            "rgb": visual_rgb,
            "depth": visual_depth,
            "fused": fused,
            "instruction": words,
            "subinstruction": sub_code,
            "subinstruction_present": sub_present,
            "history": history,
            "prev_action": prev_action.long().view(-1).clamp(0, 4),
        }
        if rgb_map is not None:
            encoded["rgb_map"] = rgb_map
            encoded["depth_map"] = depth_map
        if self.brain == "v5" and self.pretrained is not None:
            sentence, subs, sub_mask = self.pretrained.language_bank()
            encoded["sentence"] = sentence
            encoded["sub_bank"] = subs
            encoded["sub_mask"] = sub_mask
        return encoded


def _freeze(module: nn.Module, frozen: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = not frozen
