"""MLANet towers and instruction tables live in this repository."""

from pathlib import Path

from spikingnav.brain_vln.mlanet.instructions import (
    _dataset_dir,
    fit_sub_rows,
    r2r_tokens,
)
from spikingnav.brain_vln.mlanet.mlanet_policy import build_mlanet


def test_token_rows_pad_inside_the_mlanet_widths():
    assert r2r_tokens([1, 2504, 3], 2504)[:4] == [1, 1, 3, 0]
    assert len(r2r_tokens([1], 2504)) == 100
    rows = fit_sub_rows([[1, 2], [3] * 80])
    assert len(rows) == 12
    assert rows[0][:2] == [1, 2]
    assert len(rows[1]) == 77


def test_preprocessed_split_is_read_from_this_project():
    found = _dataset_dir("R2R_VLNCE_v1-3_preprocessed")
    assert found.is_relative_to(Path(__file__).resolve().parents[2])
    assert (found / "train").is_dir()


def test_fsasub_is_inside_this_project():
    found = _dataset_dir("R2R_VLNCE_FSASub")
    assert found.is_relative_to(Path(__file__).resolve().parents[2])
    assert (found / "train" / "train_sub.json.gz").is_file()


def test_policy_builder_is_local():
    assert Path(build_mlanet.__code__.co_filename).is_relative_to(Path(__file__).resolve().parents[2])
