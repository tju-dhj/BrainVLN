"""Visual encoders whose state_dict keys match official VLN-CE checkpoints."""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces
from torch import Tensor

from spikingnav.brain_vln.mlanet.ddppo_resnet import resnet50


class ResNetEncoder(nn.Module):
    """Depth backbone from Habitat DD-PPO (habitat-lab 0.1.7)."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        baseplanes: int = 32,
        ngroups: int = 16,
        make_backbone=resnet50,
        normalize_visual_inputs: bool = False,
    ) -> None:
        super().__init__()
        if "depth" in observation_space.spaces:
            self._n_input_depth = observation_space.spaces["depth"].shape[2]
            spatial_size = observation_space.spaces["depth"].shape[0] // 2
        else:
            self._n_input_depth = 0
            spatial_size = 128
        self._n_input_rgb = 0
        self.running_mean_and_var = nn.Sequential()
        if not self.is_blind:
            self.backbone = make_backbone(self._n_input_depth, baseplanes, ngroups)
            final_spatial = int(spatial_size * self.backbone.final_spatial_compress)
            num_compression_channels = int(round(2048 / (final_spatial ** 2)))
            self.compression = nn.Sequential(
                nn.Conv2d(
                    self.backbone.final_channels,
                    num_compression_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(1, num_compression_channels),
                nn.ReLU(True),
            )
            self.output_shape = (
                num_compression_channels,
                final_spatial,
                final_spatial,
            )

    @property
    def is_blind(self) -> bool:
        return self._n_input_rgb + self._n_input_depth == 0

    def forward(self, observations: Dict[str, Tensor]) -> Tensor:
        depth = observations["depth"].permute(0, 3, 1, 2)
        x = F.avg_pool2d(depth, 2)
        x = self.backbone(x)
        return self.compression(x)


class VlnResnetDepthEncoder(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Dict,
        output_size: int = 128,
        backbone: str = "resnet50",
        trainable: bool = False,
        spatial_output: bool = False,
    ) -> None:
        super().__init__()
        if backbone != "resnet50":
            raise ValueError(f"Unsupported depth backbone {backbone}")
        self.visual_encoder = ResNetEncoder(observation_space)
        for param in self.visual_encoder.parameters():
            param.requires_grad_(trainable)
        self.spatial_output = spatial_output
        if not spatial_output:
            self.output_shape = (output_size,)
            self.visual_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(int(np.prod(self.visual_encoder.output_shape)), output_size),
                nn.ReLU(True),
            )
        else:
            self.spatial_embeddings = nn.Embedding(
                self.visual_encoder.output_shape[1] * self.visual_encoder.output_shape[2],
                64,
            )
            channels = (
                self.visual_encoder.output_shape[0]
                + self.spatial_embeddings.embedding_dim
            )
            self.output_shape = (
                channels,
                self.visual_encoder.output_shape[1],
                self.visual_encoder.output_shape[2],
            )

    @property
    def is_blind(self) -> bool:
        return self.visual_encoder.is_blind

    def forward(self, observations) -> Tensor:
        x = self.visual_encoder(observations)
        if not self.spatial_output:
            return self.visual_fc(x)
        b, _, h, w = x.size()
        spatial = (
            self.spatial_embeddings(
                torch.arange(
                    self.spatial_embeddings.num_embeddings,
                    device=x.device,
                    dtype=torch.long,
                )
            )
            .view(1, -1, h, w)
            .expand(b, self.spatial_embeddings.embedding_dim, h, w)
        )
        return torch.cat([x, spatial], dim=1)


class TorchVisionResNet50(nn.Module):
    def __init__(
        self,
        output_size: int,
        normalize_visual_inputs: bool = False,
        trainable: bool = False,
        spatial_output: bool = False,
    ) -> None:
        super().__init__()
        import torchvision.models as models

        self.normalize_visual_inputs = normalize_visual_inputs
        self.spatial_output = spatial_output
        try:
            resnet = models.resnet50(weights=None)
        except TypeError:
            resnet = models.resnet50(pretrained=False)
        modules = list(resnet.children())
        self.resnet_layer_size = modules[-1].in_features
        self.cnn = nn.Sequential(*modules[:-1])
        for param in self.cnn.parameters():
            param.requires_grad_(trainable)
        self.cnn.train(trainable)
        if not spatial_output:
            self.output_shape = (output_size,)
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.resnet_layer_size, output_size),
                nn.ReLU(),
            )
        else:

            class SpatialAvgPool(nn.Module):
                def forward(self, x):
                    return F.adaptive_avg_pool2d(x, (4, 4))

            self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
            self.cnn.avgpool = SpatialAvgPool()
            self.spatial_embeddings = nn.Embedding(4 * 4, 64)
            self.output_shape = (
                self.resnet_layer_size + self.spatial_embeddings.embedding_dim,
                4,
                4,
            )

    @property
    def is_blind(self) -> bool:
        return False

    def forward(self, observations) -> Tensor:
        rgb = observations["rgb"].permute(0, 3, 1, 2).contiguous().float() / 255.0
        if self.normalize_visual_inputs:
            mean = rgb.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
            std = rgb.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
            rgb = (rgb - mean) / std
        features = self.cnn(rgb)
        if not self.spatial_output:
            return self.fc(features)
        b, _, h, w = features.size()
        spatial = (
            self.spatial_embeddings(
                torch.arange(
                    self.spatial_embeddings.num_embeddings,
                    device=features.device,
                    dtype=torch.long,
                )
            )
            .view(1, -1, h, w)
            .expand(b, self.spatial_embeddings.embedding_dim, h, w)
        )
        return torch.cat([features, spatial], dim=1)
