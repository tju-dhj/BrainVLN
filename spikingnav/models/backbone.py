"""SEW-ResNet18-style spike backbone (paper Sec. III-C1, T=4)."""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
from torchvision.models import resnet18

from spikingnav.config import DEFAULT_CONFIG
from spikingnav.nn.lif import LIF


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SEWBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        neuron_fn: Optional[Callable[[], LIF]] = None,
    ) -> None:
        super().__init__()
        if neuron_fn is None:
            neuron_fn = LIF
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.sn1 = neuron_fn()
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.sn2 = neuron_fn()
        self.downsample = downsample
        self.downsample_sn = neuron_fn() if downsample is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.sn1(self.bn1(self.conv1(x)))
        out = self.sn2(self.bn2(self.conv2(out)))
        if self.downsample is not None:
            identity = self.downsample_sn(self.downsample(x))
        return out + identity


class SEWResNet18(nn.Module):
    """SEW-ResNet18 that returns a 7x7x512 spatial map for 224 input."""

    def __init__(
        self,
        leak: float = DEFAULT_CONFIG.leak,
        threshold: float = DEFAULT_CONFIG.threshold,
        surrogate_alpha: float = DEFAULT_CONFIG.surrogate_alpha,
        pretrained_ann: bool = True,
    ) -> None:
        super().__init__()
        neuron_fn = lambda: LIF(leak, threshold, surrogate_alpha)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.sn1 = neuron_fn()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inplanes = 64
        self.layer1 = self._make_layer(64, 2, stride=1, neuron_fn=neuron_fn)
        self.layer2 = self._make_layer(128, 2, stride=2, neuron_fn=neuron_fn)
        self.layer3 = self._make_layer(256, 2, stride=2, neuron_fn=neuron_fn)
        self.layer4 = self._make_layer(512, 2, stride=2, neuron_fn=neuron_fn)
        if pretrained_ann:
            self.load_ann_imagenet()

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int,
        neuron_fn: Callable[[], LIF],
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )
        layers = [
            SEWBasicBlock(self.inplanes, planes, stride, downsample, neuron_fn)
        ]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(SEWBasicBlock(self.inplanes, planes, 1, None, neuron_fn))
        return nn.Sequential(*layers)

    def load_ann_imagenet(self) -> None:
        try:
            weights = resnet18(weights="IMAGENET1K_V1").state_dict()
        except TypeError:
            weights = resnet18(pretrained=True).state_dict()
        own = self.state_dict()
        mapped = {k: v for k, v in weights.items() if k in own and own[k].shape == v.shape}
        own.update(mapped)
        self.load_state_dict(own)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sn1(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
