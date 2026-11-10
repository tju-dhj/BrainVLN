#!/usr/bin/env python3
"""Write ~/.ai2thor/cuda-vulkan-mapping.json.

AI2-THOR aborts when vulkaninfo lists the same GPU UUID twice, which happens
when both /usr/share and /etc ship nvidia_icd.json. Keeping the first Vulkan
index for each UUID matches the device Unity should use.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path


def main() -> None:
    env = os.environ.copy()
    env.pop("DISPLAY", None)
    vulkan = subprocess.run(
        ["vulkaninfo"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env=env,
        check=True,
    )
    smi = subprocess.run(
        ["nvidia-smi", "-L"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=True,
    )

    current = None
    uuid_to_vulkan = {}
    for line in vulkan.stdout.splitlines():
        gpu_match = re.match(r"GPU(\d+):", line)
        if gpu_match:
            current = int(gpu_match.group(1))
            continue
        if "deviceUUID" not in line or current is None:
            continue
        device_uuid = line.split("=", 1)[1].strip()
        uuid_to_vulkan.setdefault(device_uuid, current)

    mapping = {}
    missing = []
    for line in smi.stdout.splitlines():
        match = re.match(r"GPU (\d+):.*\(UUID: GPU-([^)]+)\)", line)
        if match is None:
            continue
        cuda_index = int(match.group(1))
        device_uuid = match.group(2)
        vulkan_index = uuid_to_vulkan.get(device_uuid)
        if vulkan_index is None:
            missing.append(cuda_index)
            continue
        mapping[str(cuda_index)] = vulkan_index

    path = Path.home() / ".ai2thor" / "cuda-vulkan-mapping.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping))
    print(f"Wrote {path}: {mapping}")
    if missing:
        print(f"CUDA GPUs with no Vulkan device: {missing}")


if __name__ == "__main__":
    main()
