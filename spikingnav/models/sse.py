"""Spiking Sensing Encoder (paper Sec. III-C1, Eqs. 4-6)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from spikingnav.config import DEFAULT_CONFIG, SpikingNavConfig
from spikingnav.models.backbone import SEWResNet18
from spikingnav.nn.lif import LIF, reset_net


class FeatureCompressor(nn.Module):
    """Two-conv compressor: 512 -> 128 -> D_z, spatial size kept at 7x7."""

    def __init__(
        self,
        in_channels: int = 512,
        hidden_channels: int = 128,
        out_channels: int = 32,
        leak: float = 0.5,
        threshold: float = 1.0,
        surrogate_alpha: float = 2.0,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.sn1 = LIF(leak, threshold, surrogate_alpha)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.sn2 = LIF(leak, threshold, surrogate_alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sn1(self.bn1(self.conv1(x)))
        x = self.sn2(self.bn2(self.conv2(x)))
        return x


class FusionBackbone(nn.Module):
    """Two-conv fusion on [z; q] -> D_z, then flatten to D_in = H_z W_z D_z."""

    def __init__(
        self,
        in_channels: int = 64,
        hidden_channels: int = 128,
        out_channels: int = 32,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x.flatten(1)


class TargetBackbone(nn.Module):
    """ObjectNav category embedding or PointNav GPS encoder."""

    def __init__(self, goal_dims: int = 32, num_categories: Optional[int] = 12) -> None:
        super().__init__()
        self.goal_dims = goal_dims
        self.num_categories = num_categories
        if num_categories is not None:
            self.embed = nn.Embedding(num_categories, goal_dims)
        else:
            self.embed = None
        self.point_encoder = nn.Sequential(
            nn.Linear(2, goal_dims),
            nn.ReLU(inplace=True),
            nn.Linear(goal_dims, goal_dims),
        )

    def forward(self, goal: torch.Tensor) -> torch.Tensor:
        if goal.dim() == 1 or (goal.dim() == 2 and goal.shape[-1] == 1):
            goal = goal.view(-1).long()
            if self.embed is None:
                raise ValueError("Category goal received but num_categories is None")
            return self.embed(goal)
        if goal.dim() >= 2 and goal.shape[-1] == 2:
            return self.point_encoder(goal.float().view(goal.shape[0], 2))
        raise ValueError(f"Unsupported goal shape {tuple(goal.shape)}")


class SpikingSensingEncoder(nn.Module):
    """SSE: visual backbone + target backbone + fusion."""

    def __init__(
        self,
        cfg: Optional[SpikingNavConfig] = None,
        num_categories: Optional[int] = 12,
        pretrained_ann: bool = True,
    ) -> None:
        super().__init__()
        self.cfg = cfg or DEFAULT_CONFIG
        self.visual = SEWResNet18(
            leak=self.cfg.leak,
            threshold=self.cfg.threshold,
            surrogate_alpha=self.cfg.surrogate_alpha,
            pretrained_ann=pretrained_ann,
        )
        self.compressor = FeatureCompressor(
            hidden_channels=self.cfg.compressor_hidden,
            out_channels=self.cfg.visual_channels,
            leak=self.cfg.leak,
            threshold=self.cfg.threshold,
            surrogate_alpha=self.cfg.surrogate_alpha,
        )
        self.target = TargetBackbone(self.cfg.goal_dims, num_categories)
        self.fusion = FusionBackbone(
            in_channels=self.cfg.visual_channels + self.cfg.goal_dims,
            hidden_channels=self.cfg.compressor_hidden,
            out_channels=self.cfg.visual_channels,
        )
        self.out_dim = self.cfg.feature_hw * self.cfg.feature_hw * self.cfg.visual_channels

    def encode_visual(self, rgb: torch.Tensor) -> torch.Tensor:
        """rgb: [B, 3, H, W] in roughly ImageNet-normalized range."""
        reset_net(self.visual)
        reset_net(self.compressor)
        acc = None
        for _ in range(self.cfg.time_steps):
            feats = self.compressor(self.visual(rgb))
            acc = feats if acc is None else acc + feats
        return acc / float(self.cfg.time_steps)

    def forward(self, rgb: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        z = self.encode_visual(rgb)
        e = self.target(goal)
        q = e.view(e.shape[0], e.shape[1], 1, 1).expand(-1, -1, z.shape[2], z.shape[3])
        return self.fusion(torch.cat([z, q], dim=1))
