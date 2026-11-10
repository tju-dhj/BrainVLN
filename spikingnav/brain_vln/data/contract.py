"""P01 constants for collected R2R-CE episodes.

The scene split is a total order on directory names. It does not draw a
random seed; repeating the sort reproduces the same holdout.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

ACTION_NAMES: Tuple[str, ...] = ("STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT")
ACTION_PAD = -1
ALIGNMENT = "rgb[t] and depth[t] are the observation before action[t]"
FRAME_PATTERN = "{step}_0.png"
WINDOW_SIZE = 64
BURN_IN_STEPS = 16
TRUNCATE_EVERY = 32
HOLDOUT_SCENE_COUNT = 6
SPLIT_SEED = None

MODEL_INPUT_KEYS = ("rgb", "depth", "depth_valid", "tokens", "token_mask")
LABEL_KEYS = ("actions",)
TRAINER_MASK_KEYS = (
    "valid_mask",
    "loss_mask",
    "burn_in_mask",
    "episode_mask",
    "reset_mask",
    "detach_mask",
)
BOOKKEEPING_KEYS = (
    "episode_id",
    "scene",
    "instruction_text",
    "window_start",
    "episode_steps",
)
BATCH_KEYS = MODEL_INPUT_KEYS + LABEL_KEYS + TRAINER_MASK_KEYS + BOOKKEEPING_KEYS
FORBIDDEN_FIELDS = (
    "goal",
    "geodesic_distance",
    "positions",
    "locations",
    "reference_path",
    "start_position",
    "start_rotation",
)


class DataContractError(ValueError):
    """A collected episode disagrees with the P01 loading contract."""


def assign_scenes(
    scene_names: Iterable[str],
    holdout_count: int = HOLDOUT_SCENE_COUNT,
) -> Dict[str, Tuple[str, ...]]:
    """Hold out the last scenes in lexicographic directory-name order."""
    names = list(scene_names)
    if len(names) != len(set(names)):
        raise DataContractError("scene names must be unique")
    ordered = tuple(sorted(names))
    if holdout_count < 1 or holdout_count >= len(ordered):
        raise DataContractError(
            f"holdout_count={holdout_count} must leave at least one train scene "
            f"out of {len(ordered)}"
        )
    return {
        "train": ordered[:-holdout_count],
        "holdout": ordered[-holdout_count:],
    }
