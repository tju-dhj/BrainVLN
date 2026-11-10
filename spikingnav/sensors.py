"""RGB sensor wrapper that applies RobustNav visual corruptions."""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence

import numpy as np
from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor

from spikingnav.corruptions import apply_corruption_sequence


def _install_thor_gpu_override() -> None:
    """Point CloudRendering at nvidia-smi ids, not PyTorch ordinals.

    On this machine the CUDA runtime only opens four devices, and their
    ordinals are not nvidia-smi indices: ordinal 1 is nvidia-smi GPU 2.
    AI2-THOR then rewrites ``gpu_device`` through ``CUDA_VISIBLE_DEVICES``
    and uses that integer as an nvidia-smi index. ``SPIKINGNAV_THOR_GPU_IDS``
    is the nvidia-smi id for each local training device, in the same order
    as ``SPIKINGNAV_GPU_IDS``.
    """
    raw = os.environ.get("SPIKINGNAV_THOR_GPU_IDS", "").strip()
    if not raw:
        return
    nvidia_ids = [int(part) for part in raw.split(",") if part.strip()]
    from ai2thor.controller import Controller

    original = Controller.unity_command
    if getattr(original, "_spikingnav_patched", False):
        return

    def patched(self, width, height, headless):
        # Controller.__init__ rewrites gpu_device through CUDA_VISIBLE_DEVICES
        # and then starts Unity before the constructor returns. Translate that
        # rewritten id back to the nvidia-smi id while the command is built.
        if self.gpu_device is not None:
            visible = [
                int(part)
                for part in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
                if part.strip()
            ]
            try:
                local_index = visible.index(int(self.gpu_device))
            except ValueError:
                local_index = None
            if local_index is not None and local_index < len(nvidia_ids):
                self.gpu_device = nvidia_ids[local_index]
        return original(self, width, height, headless)

    patched._spikingnav_patched = True  # type: ignore[attr-defined]
    Controller.unity_command = patched


_install_thor_gpu_override()


def maybe_corrupt_frame(
    frame: np.ndarray,
    corruptions: Optional[Sequence[str]] = None,
    severities: Optional[Sequence[int]] = None,
) -> np.ndarray:
    if not corruptions:
        return frame
    return apply_corruption_sequence(frame, corruptions, severities)


class CorruptibleRGBSensorThor(RGBSensorThor):
    def __init__(
        self,
        corruptions: Optional[Sequence[str]] = None,
        severities: Optional[Sequence[int]] = None,
        **kwargs: Any,
    ) -> None:
        self._sn_corruptions = list(corruptions or [])
        self._sn_severities = list(severities or [5] * len(self._sn_corruptions))
        super().__init__(**kwargs)

    def frame_from_env(self, env, task):  # type: ignore[override]
        frame = super().frame_from_env(env, task)
        return maybe_corrupt_frame(frame, self._sn_corruptions, self._sn_severities)


def make_rgb_sensor(
    height: int,
    width: int,
    uuid: str = "rgb_lowres",
    corruptions: Optional[Sequence[str]] = None,
    severities: Optional[Sequence[int]] = None,
    use_resnet_normalization: bool = True,
):
    return CorruptibleRGBSensorThor(
        height=height,
        width=width,
        use_resnet_normalization=use_resnet_normalization,
        uuid=uuid,
        corruptions=corruptions,
        severities=severities,
    )
