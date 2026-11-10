#!/usr/bin/env python3
"""BrainVLN_v5 imitation. Edit configs/brain_vln_il.yaml.

This entry does not change the v2 command line in scripts/brain_vln/train_il.py.
"""

from __future__ import annotations

import hydra
from omegaconf import OmegaConf

from spikingnav.brain_vln.il_config import settings_from_config
from train_il import train


@hydra.main(version_base="1.3", config_path="../../configs", config_name="brain_vln_il")
def main(cfg) -> None:
    args = settings_from_config(cfg)
    resolved = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    if "hydra" in resolved:
        del resolved["hydra"]
    args.config_file = "configs/brain_vln_il.yaml"
    args.resolved_yaml = OmegaConf.to_yaml(resolved)
    train(args)


if __name__ == "__main__":
    main()
