"""Contiguous episode windows for burn-in and truncated BPTT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from spikingnav.brain_vln.data.contract import BURN_IN_STEPS, WINDOW_SIZE, DataContractError


@dataclass(frozen=True)
class WindowSpec:
    """One contiguous slice of a single episode.

    ``start`` is the episode step loaded at local time 0. ``burn_in_steps``
    counts local steps that rebuild reservoir state and stay out of the loss.
    A window that begins at step 0 uses burn-in 0 and resets from a clear state,
    so those early actions remain supervised.
    """

    start: int
    valid_length: int
    burn_in_steps: int


def plan_windows(
    num_steps: int,
    window_size: int = WINDOW_SIZE,
    burn_in: int = BURN_IN_STEPS,
) -> List[WindowSpec]:
    """Cover every action with exactly one supervised step.

    Later windows overlap the previous window by ``burn_in`` steps. That
    overlap is replayed from a reset and is excluded from the loss. Requires
    ``2 * burn_in <= window_size`` so the replay cannot re-supervise actions
    already covered by the previous window.
    """
    if window_size < 1:
        raise DataContractError("window_size must be positive")
    if burn_in < 0 or burn_in * 2 > window_size:
        raise DataContractError("burn_in must satisfy 0 <= 2 * burn_in <= window_size")
    if num_steps < 1:
        raise DataContractError("episode has no steps")
    if num_steps <= window_size:
        return [WindowSpec(start=0, valid_length=num_steps, burn_in_steps=0)]

    windows: List[WindowSpec] = []
    start = 0
    while True:
        valid_length = min(window_size, num_steps - start)
        burn = 0 if start == 0 else burn_in
        if burn >= valid_length:
            raise DataContractError("window produced an empty supervised span")
        windows.append(WindowSpec(start=start, valid_length=valid_length, burn_in_steps=burn))
        if start + valid_length >= num_steps:
            return windows
        next_start = start + valid_length - burn_in
        if next_start <= start:
            raise DataContractError("window sampler failed to advance")
        start = next_start
