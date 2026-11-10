"""MLANet imitation schedule used for EnvDrop and the following DAgger tune.

The numbers are the released files ``mla_aug.yaml`` and ``mla_aug_da_tune.yaml``.
EnvDrop is one teacher-forcing pass (``p = 1``) over ``joint_train_envdrop``.
DAgger then aggregates a fresh dataset: each iteration appends ``update_size``
episodes and trains ``epochs`` passes. With ``load_from_ckpt``, the expert
probability on iteration ``i`` is ``p ** (i + 1)``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

ENVDROP_EPOCHS = 45
ENVDROP_BATCH_SIZE = 5
ENVDROP_LR = 2.5e-4
ENVDROP_ITERATIONS = 1
ENVDROP_UPDATE_SIZE = 157232
ENVDROP_P = 1.0

DAGGER_EPOCHS = 4
DAGGER_BATCH_SIZE = 5
DAGGER_LR = 2.5e-4
DAGGER_ITERATIONS = 10
DAGGER_UPDATE_SIZE = 5000
DAGGER_P = 0.5
INFLECTION_COEF = 3.2

# BrainVLN v2 reserves action id 4 for "no previous action". MLANet's first
# prev-action is 0 because its embedding has no start id. Keeping 4 matches
# the checkpoint this run continues from.
START_ACTION = 4


def dagger_beta(p: float, data_it: int) -> float:
    """Expert probability used by MLANet's DAgger collector."""
    if p > 1.0:
        return float(p - 1.0)
    if p == 0.0:
        return 0.0
    return float(p**data_it)


def inflection_weights(actions: torch.Tensor, coef: float = INFLECTION_COEF) -> torch.Tensor:
    """Per-step weights. The first label is an inflection, then every action change.

    ``actions`` is ``[T]`` without padding. This is the index MLANet builds with
    ``cat([1, oracle[1:] != oracle[:-1]])`` into ``[1.0, coef]``.
    """
    if actions.dim() != 1:
        raise ValueError("inflection weights expect one trajectory")
    change = torch.ones(actions.shape[0], dtype=torch.long, device=actions.device)
    if actions.shape[0] > 1:
        change[1:] = (actions[1:] != actions[:-1]).long()
    table = torch.tensor([1.0, float(coef)], dtype=torch.float, device=actions.device)
    return table[change]


def weighted_action_loss(logits: torch.Tensor, actions: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Mean over trajectories of the inflection-weighted step cross-entropy.

    Matches ``BaseVLNCETrainer._update_agent``: pad positions carry weight 0
    and do not enter the per-trajectory denominator.
    """
    if logits.dim() != 3:
        raise ValueError("logits must be [T, B, C]")
    per_step = F.cross_entropy(
        logits.permute(0, 2, 1),
        actions.long().clamp(min=0),
        reduction="none",
    )
    denominator = weights.sum(dim=0).clamp(min=1.0)
    return ((weights * per_step).sum(dim=0) / denominator).mean()


def select_dagger_episodes(demos: list, iteration: int, seed: int, limit: int) -> list:
    """Same seeded draw every worker uses for one DAgger iteration."""
    import random

    rng = random.Random(int(seed) + int(iteration))
    pool = list(demos)
    rng.shuffle(pool)
    count = DAGGER_UPDATE_SIZE if int(limit) <= 0 else min(int(limit), DAGGER_UPDATE_SIZE)
    return pool[:count]


def partition_episodes(demos: list, part: int, parts: int) -> list:
    """Split one scene shard into disjoint episode slices.

    Scene assignment stays intact. ``parts == 1`` returns every episode.
    """
    part = int(part)
    parts = int(parts)
    if parts < 1 or not 0 <= part < parts:
        raise ValueError(f"invalid episode part {part}/{parts}")
    if parts == 1:
        return list(demos)
    return [demo for index, demo in enumerate(demos) if index % parts == part]


def shard_dagger_demos(demos: list, num_shards: int, shard_index: int) -> list:
    """Give each worker whole scenes so a mesh is loaded by one process."""
    from spikingnav.vlnce.sim import shard_scenes

    grouped = {}
    for demo in demos:
        grouped.setdefault(demo["scene_id"], []).append(demo)
    chosen = set(shard_scenes(grouped, num_shards, shard_index))
    return [demo for demo in demos if demo["scene_id"] in chosen]
