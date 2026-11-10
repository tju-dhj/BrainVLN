"""Offline teacher forcing on stored R2R-CE expert rollouts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

from spikingnav.vlnce.constants import (
    GRAD_CLIP,
    IL_BATCH_SIZE,
    IL_EPOCHS,
    IL_LR,
    INFLECTION_COEF,
)
from spikingnav.vlnce.episodes import token_length
from spikingnav.vlnce.policy import SpikingVLNPolicy
from spikingnav.vlnce.rgb import prepare_rgb
from spikingnav.vlnce.store import find_rollouts, load_rollout


def inflection_weights(actions: torch.Tensor, coef: float = INFLECTION_COEF) -> torch.Tensor:
    """VLN-CE weight: ``coef`` on the first step and whenever the action changes."""
    squeeze = actions.dim() == 1
    if squeeze:
        actions = actions.unsqueeze(1)
    first = torch.ones(1, actions.shape[1], dtype=torch.bool, device=actions.device)
    changed = actions[1:] != actions[:-1]
    inflections = torch.cat([first, changed], 0)
    weights = torch.ones(actions.shape, dtype=torch.float32, device=actions.device)
    weights = torch.where(inflections, weights.new_full((), float(coef)).expand_as(weights), weights)
    return weights.squeeze(1) if squeeze else weights


def imitation_loss(
    logits: torch.Tensor,
    expert: torch.Tensor,
    valid: torch.Tensor,
    weights: torch.Tensor = None,
) -> torch.Tensor:
    """Weighted cross-entropy. ``logits`` is ``[T, N, A]``."""
    chosen = F.log_softmax(logits, dim=-1).gather(-1, expert.long().clamp(min=0).unsqueeze(-1))
    chosen = chosen.squeeze(-1)
    mask = valid.float()
    if weights is not None:
        mask = mask * weights.float()
    return -(chosen * mask).sum() / mask.sum().clamp(min=1.0)


class RolloutDataset(Dataset):
    def __init__(self, root: str, max_steps: int = 0, limit: int = 0) -> None:
        self.files = find_rollouts(root)
        if limit > 0:
            self.files = self.files[:limit]
        if not self.files:
            raise FileNotFoundError(f"No completed episode directories in {root}")
        self.max_steps = int(max_steps)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict:
        record = load_rollout(self.files[index])
        steps = len(record["actions"])
        if self.max_steps > 0:
            steps = min(steps, self.max_steps)
        tokens = record["tokens"]
        return {
            "rgb": record["rgb"][:steps],
            "actions": record["actions"][:steps].astype(np.int64),
            "tokens": np.asarray(tokens, dtype=np.int64),
            "token_length": token_length(tokens),
            "episode_id": record["episode_id"],
        }


def collate_rollouts(items: Sequence[dict]) -> dict:
    steps = max(len(item["actions"]) for item in items)
    length = max(int(item["token_length"]) for item in items)
    batch = len(items)
    height, width = items[0]["rgb"].shape[1:3]
    rgb = np.zeros((steps, batch, height, width, 3), dtype=np.uint8)
    actions = np.zeros((steps, batch), dtype=np.int64)
    valid = np.zeros((steps, batch), dtype=np.float32)
    weights = np.zeros((steps, batch), dtype=np.float32)
    tokens = np.zeros((batch, length), dtype=np.int64)
    lengths = np.zeros(batch, dtype=np.int64)
    for index, item in enumerate(items):
        count = len(item["actions"])
        rgb[:count, index] = item["rgb"][:count]
        actions[:count, index] = item["actions"]
        valid[:count, index] = 1.0
        step_weights = inflection_weights(torch.as_tensor(item["actions"])).numpy()
        weights[:count, index] = step_weights
        keep = int(item["token_length"])
        tokens[index, :keep] = item["tokens"][:keep]
        lengths[index] = keep
    return {
        "rgb": torch.from_numpy(rgb),
        "actions": torch.from_numpy(actions),
        "valid": torch.from_numpy(valid),
        "weights": torch.from_numpy(weights),
        "tokens": torch.from_numpy(tokens),
        "lengths": torch.from_numpy(lengths),
    }


def train_offline(
    data: str,
    embeddings,
    output_dir: str,
    epochs: int = IL_EPOCHS,
    batch_size: int = IL_BATCH_SIZE,
    lr: float = IL_LR,
    inflection: float = INFLECTION_COEF,
    max_steps: int = 0,
    pretrained_ann: bool = True,
    seed: int = 0,
    limit: int = 0,
    device: str = "",
) -> Path:
    torch.manual_seed(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    chosen = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = RolloutDataset(data, max_steps=max_steps, limit=limit)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_rollouts,
    )
    policy = SpikingVLNPolicy(embeddings, pretrained_ann=pretrained_ann).to(chosen)
    optimizer = torch.optim.Adam(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=lr,
    )
    log_path = output / "train_log.jsonl"
    history: List[dict] = []
    for epoch in range(epochs):
        policy.train()
        total = 0.0
        seen = 0
        for batch in loader:
            rgb = prepare_rgb(batch["rgb"]).to(chosen)
            tokens = batch["tokens"].to(chosen)
            lengths = batch["lengths"].to(chosen)
            expert = batch["actions"].to(chosen)
            valid = batch["valid"].to(chosen)
            weights = batch["weights"].to(chosen)
            if inflection != INFLECTION_COEF:
                weights = inflection_weights(expert, inflection) * valid
            steps, batch_size_now = expert.shape
            masks = rgb.new_ones(steps, batch_size_now, 1)
            masks[0] = 0
            logits, _values, _memory = policy(rgb, tokens, lengths, masks=masks)
            loss = imitation_loss(logits, expert, valid, weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(policy.parameters(), GRAD_CLIP)
            optimizer.step()
            total += float(loss.detach()) * int(valid.sum())
            seen += int(valid.sum())
        row = {"epoch": epoch, "loss": total / max(seen, 1), "steps": seen, "device": str(chosen)}
        history.append(row)
        with log_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(
            f"epoch {epoch} loss {row['loss']:.4f} supervised_steps {seen}",
            flush=True,
        )
        torch.save(_checkpoint(policy, embeddings, epoch, row), output / f"ckpt_epoch_{epoch}.pt")
    torch.save(_checkpoint(policy, embeddings, epochs - 1, history[-1]), output / "ckpt_last.pt")
    return output


def _checkpoint(policy, embeddings, epoch: int, row: dict) -> dict:
    return {
        "model": policy.state_dict(),
        "embeddings": embeddings.detach().cpu(),
        "epoch": epoch,
        "metrics": row,
        "task": "r2rce_offline_il",
    }


def load_policy(checkpoint_path: str, device: torch.device) -> SpikingVLNPolicy:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy = SpikingVLNPolicy(checkpoint["embeddings"], pretrained_ann=False)
    policy.load_state_dict(checkpoint["model"])
    policy.to(device)
    policy.eval()
    return policy
