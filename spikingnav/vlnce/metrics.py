"""R2R-CE path metrics that do not need the simulator."""

from __future__ import annotations

import numpy as np

from spikingnav.vlnce.constants import SUCCESS_DISTANCE


def path_length(positions) -> float:
    points = np.asarray(positions, dtype=np.float64)
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def ndtw(prediction, reference, success_distance: float = SUCCESS_DISTANCE) -> float:
    """Normalized DTW used by VLN-CE. Identical paths score 1."""
    pred = np.asarray(prediction, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if pred.ndim != 2 or ref.ndim != 2 or len(pred) == 0 or len(ref) == 0:
        return 0.0
    distance = np.linalg.norm(pred[:, None, :] - ref[None, :, :], axis=-1)
    rows, cols = distance.shape
    table = np.full((rows + 1, cols + 1), np.inf)
    table[0, 0] = 0.0
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            table[i, j] = distance[i - 1, j - 1] + min(
                table[i - 1, j], table[i, j - 1], table[i - 1, j - 1]
            )
    return float(np.exp(-table[rows, cols] / (cols * success_distance)))


def spl(success: bool, geodesic: float, traveled: float) -> float:
    if geodesic < 0:
        return 0.0
    return float(success) * geodesic / max(geodesic, traveled, 1e-6)
