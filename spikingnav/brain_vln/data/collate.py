"""Pad episode fragments into a time-major batch."""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch

from spikingnav.brain_vln.data.contract import ACTION_PAD, BATCH_KEYS, DataContractError
from spikingnav.brain_vln.data.episode import EpisodeRecord


def _pad_time(tensor: torch.Tensor, length: int, value) -> torch.Tensor:
    if tensor.shape[0] == length:
        return tensor
    if tensor.shape[0] > length:
        raise DataContractError("sample is longer than the collate target")
    padding = tensor.new_full((length - tensor.shape[0], *tensor.shape[1:]), value)
    return torch.cat((tensor, padding), dim=0)


def _pad_tokens(tokens: torch.Tensor, mask: torch.Tensor, length: int) -> tuple:
    if tokens.shape[0] == length:
        return tokens, mask
    token_pad = tokens.new_full((length - tokens.shape[0],), 0)
    mask_pad = mask.new_full((length - mask.shape[0],), False)
    return torch.cat((tokens, token_pad), dim=0), torch.cat((mask, mask_pad), dim=0)


def collate_fn(batch: Sequence[EpisodeRecord]) -> Dict[str, object]:
    """Stack samples as ``[T, B, ...]``. Each column resets its own reservoir."""
    if not batch:
        raise DataContractError("cannot collate an empty batch")
    spatial = {tuple(sample.rgb.shape[-2:]) for sample in batch}
    if len(spatial) != 1:
        raise DataContractError(f"batch mixes RGB spatial sizes {sorted(spatial)}")
    time = max(sample.rgb.shape[0] for sample in batch)
    token_length = max(sample.tokens.shape[0] for sample in batch)

    def timed(samples: List[torch.Tensor], fill) -> torch.Tensor:
        return torch.stack([_pad_time(sample, time, fill) for sample in samples], dim=1)

    token_rows = []
    mask_rows = []
    for sample in batch:
        tokens, mask = _pad_tokens(sample.tokens, sample.token_mask, token_length)
        token_rows.append(tokens)
        mask_rows.append(mask)

    collated = {
        "rgb": timed([sample.rgb for sample in batch], 0),
        "depth": timed([sample.depth for sample in batch], 0),
        "depth_valid": timed([sample.depth_valid for sample in batch], False),
        "tokens": torch.stack(token_rows, dim=0),
        "token_mask": torch.stack(mask_rows, dim=0),
        "actions": timed([sample.actions for sample in batch], ACTION_PAD),
        "valid_mask": timed([sample.valid_mask for sample in batch], False),
        "loss_mask": timed([sample.loss_mask for sample in batch], False),
        "burn_in_mask": timed([sample.burn_in_mask for sample in batch], False),
        "episode_mask": timed([sample.episode_mask for sample in batch], False),
        "reset_mask": timed([sample.reset_mask for sample in batch], False),
        "detach_mask": timed([sample.detach_mask for sample in batch], False),
        "episode_id": [sample.episode_id for sample in batch],
        "scene": [sample.scene for sample in batch],
        "instruction_text": [sample.instruction_text for sample in batch],
        "window_start": torch.tensor([sample.window_start for sample in batch], dtype=torch.long),
        "episode_steps": torch.tensor([sample.episode_steps for sample in batch], dtype=torch.long),
    }
    if tuple(collated) != BATCH_KEYS:
        raise DataContractError("collate output drifted from the batch contract")
    return collated
