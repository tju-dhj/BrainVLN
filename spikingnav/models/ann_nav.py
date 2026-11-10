"""Matched ANNNav baseline: ResNet18 + GRU (RobustNav / AllenAct)."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torchvision.models import resnet18

from spikingnav.config import DEFAULT_CONFIG, SpikingNavConfig
from spikingnav.models.actor_critic import _to_nchw, flatten_leading
from spikingnav.models.sse import TargetBackbone


class ResNet18Encoder(nn.Module):
    """Spatial ResNet18 encoder producing 7x7x512 maps."""

    def __init__(self, pretrained: bool = True, freeze: bool = True) -> None:
        super().__init__()
        try:
            net = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        except TypeError:
            net = resnet18(pretrained=pretrained)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ANNNavModel(nn.Module):
    """ResNet18 visual encoder + goal fusion + GRU policy."""

    def __init__(
        self,
        action_dim: int,
        num_categories: Optional[int] = 12,
        cfg: Optional[SpikingNavConfig] = None,
        freeze_backbone: bool = True,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.cfg = cfg or DEFAULT_CONFIG
        hidden = self.cfg.hidden_size
        self.visual = ResNet18Encoder(pretrained=pretrained, freeze=freeze_backbone)
        self.compressor = nn.Sequential(
            nn.Conv2d(512, self.cfg.compressor_hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.cfg.compressor_hidden, self.cfg.visual_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.target = TargetBackbone(self.cfg.goal_dims, num_categories)
        self.fusion = nn.Sequential(
            nn.Conv2d(
                self.cfg.visual_channels + self.cfg.goal_dims,
                self.cfg.compressor_hidden,
                3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.cfg.compressor_hidden, self.cfg.visual_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out_dim = self.cfg.feature_hw * self.cfg.feature_hw * self.cfg.visual_channels
        self.gru = nn.GRU(self.out_dim, hidden, batch_first=False)
        self.actor = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)

    def encode(self, rgb: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        z = self.compressor(self.visual(_to_nchw(rgb)))
        e = self.target(goal)
        q = e.view(e.shape[0], e.shape[1], 1, 1).expand(-1, -1, z.shape[2], z.shape[3])
        return self.fusion(torch.cat([z, q], dim=1)).flatten(1)

    def initial_memory(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.cfg.hidden_size, device=device)

    def forward(
        self,
        rgb: torch.Tensor,
        goal: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if rgb.dim() == 4:
            rgb = rgb.unsqueeze(0)
            if goal.dim() in (1, 2):
                goal = goal.unsqueeze(0)
        t_steps, batch = rgb.shape[:2]
        if memory is None:
            memory = self.initial_memory(batch, rgb.device)
        if masks is None:
            masks = rgb.new_ones(t_steps, batch, 1)

        r_list = []
        for t in range(t_steps):
            goal_t = goal[t] if goal.shape[0] == t_steps else goal
            r_list.append(self.encode(rgb[t], goal_t))
        r = torch.stack(r_list, 0)

        outputs = []
        h = memory
        for t in range(t_steps):
            h = h * masks[t].view(1, batch, 1)
            out, h = self.gru(r[t].unsqueeze(0), h)
            outputs.append(out.squeeze(0))
        hidden = torch.stack(outputs, 0)
        return self.actor(hidden), self.critic(hidden), h


class ANNNavActorCritic:
    @staticmethod
    def build(
        action_space,
        observation_space,
        goal_sensor_uuid: str,
        rgb_uuid: str = "rgb_lowres",
        num_categories: Optional[int] = 12,
        cfg: Optional[SpikingNavConfig] = None,
        freeze_backbone: bool = True,
    ):
        from allenact.algorithms.onpolicy_sync.policy import ActorCriticModel
        from allenact.base_abstractions.distributions import CategoricalDistr
        from allenact.base_abstractions.misc import ActorCriticOutput, Memory

        cfg = cfg or DEFAULT_CONFIG

        class _Model(ActorCriticModel[CategoricalDistr]):
            def __init__(self) -> None:
                super().__init__(
                    action_space=action_space, observation_space=observation_space
                )
                self.goal_sensor_uuid = goal_sensor_uuid
                self.rgb_uuid = rgb_uuid
                self.core = ANNNavModel(
                    action_dim=action_space.n,
                    num_categories=num_categories,
                    cfg=cfg,
                    freeze_backbone=freeze_backbone,
                )
                self.memory_key = "rnn"

            @property
            def recurrent_hidden_state_size(self) -> int:
                return cfg.hidden_size

            @property
            def num_recurrent_layers(self) -> int:
                return 1

            def _recurrent_memory_specification(self):
                return {
                    self.memory_key: (
                        (
                            ("layer", 1),
                            ("sampler", None),
                            ("hidden", cfg.hidden_size),
                        ),
                        torch.float32,
                    )
                }

            def forward(  # type: ignore[override]
                self,
                observations: Dict[str, torch.Tensor],
                memory: Memory,
                prev_actions: torch.Tensor,
                masks: torch.FloatTensor,
            ):
                rgb = observations[self.rgb_uuid]
                goal = observations[self.goal_sensor_uuid]
                nsteps, nsamplers = masks.shape[:2]
                rgb_flat, _ = flatten_leading(rgb, 2)
                goal_flat, _ = flatten_leading(goal, 2)
                r = self.core.encode(rgb_flat, goal_flat).view(nsteps, nsamplers, -1)
                h = memory.tensor(self.memory_key)
                logits = []
                values = []
                for t in range(nsteps):
                    h = h * masks[t].view(1, nsamplers, 1)
                    out, h = self.core.gru(r[t].unsqueeze(0), h)
                    hid = out.squeeze(0)
                    logits.append(self.core.actor(hid))
                    values.append(self.core.critic(hid))
                mem = memory.set_tensor(self.memory_key, h)
                return (
                    ActorCriticOutput(
                        distributions=CategoricalDistr(logits=torch.stack(logits, 0)),
                        values=torch.stack(values, 0),
                        extras={},
                    ),
                    mem,
                )

        return _Model()
