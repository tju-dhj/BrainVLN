import numpy as np

from spikingnav.config import VISUAL_CORRUPTIONS
from spikingnav.corruptions import apply_corruption, apply_corruption_sequence, canonicalize


def test_each_corruption_preserves_shape():
    image = np.random.randint(0, 255, (64, 80, 3), dtype=np.uint8)
    for name in VISUAL_CORRUPTIONS:
        out = apply_corruption(image, name, severity=3)
        assert out.shape == image.shape
        assert out.dtype == np.uint8


def test_alias_and_sequence():
    image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    assert canonicalize("defocus_blur") == "Defocus Blur"
    out = apply_corruption_sequence(image, ["Speckle Noise", "Low Lighting"], [2, 4])
    assert out.shape == image.shape
