"""Hydra config for BrainVLN imitation."""

from pathlib import Path

import pytest

hydra = pytest.importorskip("hydra")
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from spikingnav.brain_vln.il_config import settings_from_config

ROOT = Path(__file__).resolve().parents[2]


def setup_function():
    GlobalHydra.instance().clear()


def teardown_function():
    GlobalHydra.instance().clear()


def _compose(overrides=None):
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        return compose(config_name="brain_vln_il", overrides=overrides or [])


def test_il_config_is_the_v5_run():
    args = settings_from_config(_compose())
    assert args.output == "storage/brain_vln/external_ssd1/BrainVLN_v5"
    assert args.root == "storage/collect_data/train"
    assert args.brain == "v5"
    assert args.variant == "B4"
    assert args.layout == "cells"
    assert args.encoder == "mlanet"
    assert args.stdp is False
    assert args.epochs == 100
    assert args.batch_size == 128
    assert args.microbatch == 32
    assert args.workers == 2
    assert args.lr == 1.0e-3
    assert args.grad_clip == 1.0
    assert args.seed == 0
    assert args.checkpoint_every == 200
    assert args.data_parallel is False
    assert args.resume == ""
    assert args.episodic_tau == 30.0
    assert args.stop_neurons == 64
    assert args.view_change_threshold == 0.05


def test_il_config_accepts_a_command_override():
    args = settings_from_config(_compose(["epochs=1", "batch_size=2", "episodic_tau=20"]))
    assert args.epochs == 1
    assert args.batch_size == 2
    assert args.episodic_tau == 20.0


def test_il_config_rejects_an_unknown_key():
    cfg = OmegaConf.create(OmegaConf.to_container(_compose(), resolve=True))
    cfg.extra = 1
    with pytest.raises(ValueError, match="unknown extra"):
        settings_from_config(cfg)


def test_il_config_rejects_v5_with_stdp():
    with pytest.raises(ValueError, match="BrainVLN_v5"):
        settings_from_config(_compose(["stdp=true"]))
