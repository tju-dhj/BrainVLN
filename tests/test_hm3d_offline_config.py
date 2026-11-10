from pathlib import Path

import pytest

hydra = pytest.importorskip("hydra")
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from scripts.train_hm3d_offline_experiment import namespace_from_config


ROOT = Path(__file__).resolve().parents[1]


def setup_function():
    GlobalHydra.instance().clear()


def teardown_function():
    GlobalHydra.instance().clear()


def test_offline_il_config_uses_the_current_training_defaults():
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="hm3d_offline_il", overrides=["data=train.json", "output=storage/run"])
    args = namespace_from_config(cfg)
    assert args.data == "train.json"
    assert args.batch_size == 32
    assert args.save_every == 100
    assert args.epochs == 0
    assert args.transition_c == 3.0
    assert args.normalization == "frozen_bn"
    assert args.fixed is False
    assert args.student_eval_list == ""


def test_offline_il_config_rejects_an_unknown_normalization():
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(
            config_name="hm3d_offline_il",
            overrides=["data=train.json", "output=storage/run", "normalization=layernorm"],
        )
    with pytest.raises(ValueError, match="normalization"):
        namespace_from_config(cfg)
