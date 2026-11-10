#!/usr/bin/env python3
"""Train offline HM3D IL from configs/hm3d_offline_il.yaml."""
from types import SimpleNamespace

import hydra
from omegaconf import OmegaConf

from spikingnav.habitat_objectnav.offline_experiment import train

_CHOICES = {
    "class_balance": ("none", "balanced", "sqrt_balanced"),
    "normalization": ("frozen_bn", "adaptive_bn", "group_norm"),
    "spn_stability": ("none", "bounded"),
}


def namespace_from_config(cfg):
    values = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(values, dict):
        raise TypeError("Training config must be a mapping")
    values.pop("hydra", None)
    missing = [key for key in ("data", "output") if not values.get(key)]
    if missing:
        raise ValueError("Set " + ", ".join(missing) + " in configs/hm3d_offline_il.yaml")
    args = SimpleNamespace(**values)
    if args.resume and args.init_checkpoint:
        raise ValueError("resume and init_checkpoint are mutually exclusive")
    if args.batch_size < 1 or args.updates < 1 or args.save_every < 1 or args.log_every < 1:
        raise ValueError("batch_size, updates, save_every and log_every must be positive")
    if args.epochs < 0:
        raise ValueError("epochs must be non-negative")
    if not args.transition_c > 0:
        raise ValueError("transition_c must be positive")
    if args.keep_sr_checkpoints < 1:
        raise ValueError("keep_sr_checkpoints must be positive")
    for key, choices in _CHOICES.items():
        if getattr(args, key) not in choices:
            raise ValueError(f"{key} must be one of {choices}")
    return args


@hydra.main(version_base="1.3", config_path="../configs", config_name="hm3d_offline_il")
def main(cfg):
    train(namespace_from_config(cfg))


if __name__ == "__main__":
    main()
