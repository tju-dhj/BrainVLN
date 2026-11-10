"""BrainVLN_v2 keeps one current per grid cell and channel."""

import torch

from spikingnav.brain_vln.config import tiny_config
from spikingnav.brain_vln.model import (
    CELL_CHANNELS,
    DEPTH_CELL_COUNT,
    DEPTH_GRID,
    RGB_CELL_COUNT,
    RGB_GRID,
    SPATIAL_CELL_COUNT,
    BrainVLNModel,
    cell_currents,
)


def test_cell_counts_match_the_grids():
    assert RGB_GRID == (3, 3)
    assert DEPTH_GRID == (4, 4)
    assert RGB_CELL_COUNT == 3 * 3 * CELL_CHANNELS
    assert DEPTH_CELL_COUNT == 4 * 4 * CELL_CHANNELS
    assert SPATIAL_CELL_COUNT == 800


def test_each_cell_is_normalized_on_its_own():
    left = torch.tensor([1.0, 3.0, 5.0, 7.0])
    right = left * 10
    feature_map = torch.zeros(1, 4, 1, 2)
    feature_map[0, :, 0, 0] = left
    feature_map[0, :, 0, 1] = right
    current = cell_currents(feature_map, scale=1.0).reshape(1, 2, 4)[0]
    assert current.shape == (2, 4)
    assert torch.allclose(current[0].mean(), torch.zeros(()), atol=1e-5)
    assert torch.allclose(current[1].mean(), torch.zeros(()), atol=1e-5)
    assert torch.allclose(current[0], current[1], atol=1e-5)


def test_cell_order_is_row_column_channel():
    feature_map = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(1, 2, 3, 4)
    current = cell_currents(feature_map, scale=1.0)
    assert current.shape == (1, 3 * 4 * 2)
    raw = feature_map.permute(0, 2, 3, 1).reshape(1, -1, 2)
    expected = torch.nn.functional.layer_norm(raw, (2,)).reshape(1, -1)
    assert torch.allclose(current, expected)


def test_cell_layout_rejects_the_pooled_population_size():
    config = tiny_config(encoder="mlanet", visual_layout="cells", variant="B4")
    try:
        BrainVLNModel(config)
    except ValueError as exc:
        assert "800" in str(exc)
    else:
        raise AssertionError("cell layout accepted a spatial population other than 800")
