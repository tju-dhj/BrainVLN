"""One optimization step and checkpoint IO for offline imitation."""

from __future__ import annotations

import gc
from pathlib import Path

import torch

from spikingnav.brain_vln.losses import total_loss
from spikingnav.brain_vln.model import BrainVLNModel

# One optimizer step still covers the whole dataloader batch. A 32GB GPU cannot
# hold 128 windows of the 64-step reservoir graph at once, so the batch is
# split only for the forward and backward. The loss is the same masked mean.
_MICROBATCH = 64
_TIME_MAJOR = {
    "rgb",
    "depth",
    "depth_valid",
    "actions",
    "valid_mask",
    "loss_mask",
    "burn_in_mask",
    "episode_mask",
    "reset_mask",
    "detach_mask",
}
_BATCH_MAJOR = {"tokens", "token_mask", "window_start", "episode_steps"}


def _slice_batch(batch: dict, start: int, stop: int) -> dict:
    sliced = {}
    for key, value in batch.items():
        if key in _TIME_MAJOR:
            sliced[key] = value[:, start:stop]
        elif key in _BATCH_MAJOR:
            sliced[key] = value[start:stop]
        elif isinstance(value, list):
            sliced[key] = value[start:stop]
        else:
            sliced[key] = value
    return sliced


def train_batch(
    model: BrainVLNModel,
    batch: dict,
    optimizer,
    grad_clip: float = 1.0,
    state=None,
    microbatch: int = _MICROBATCH,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    width = int(batch["rgb"].shape[1])
    total_count = int(batch["loss_mask"].sum().item())
    weighted_loss = 0.0
    correct = 0
    last_state = None
    if microbatch < 1:
        raise ValueError("microbatch must be positive")
    for start in range(0, width, microbatch):
        stop = min(start + microbatch, width)
        part = _slice_batch(batch, start, stop)
        part_state = None if state is None else state.select(
            torch.arange(start, stop, device=state.membrane.device)
        )
        # Online STDP mutates weights in the first forward. Disable it for the
        # checkpoint recompute on backward so the same update is not applied twice.
        # DataParallel stores the flag on the wrapped module.
        core = model.module if hasattr(model, "module") else model
        core.allow_stdp = True
        logits, step_state, aux = model(
            part["rgb"],
            part["depth"],
            part["depth_valid"],
            part["tokens"],
            part["token_mask"],
            part["valid_mask"],
            part["reset_mask"],
            part["detach_mask"],
            actions=part["actions"],
            state=part_state,
            instruction_text=part.get("instruction_text"),
        )
        core.allow_stdp = False
        progress_target = None
        if core.config.progress_weight > 0:
            steps = part["episode_steps"].to(logits.dtype).clamp(min=1)
            time = torch.arange(logits.shape[0], device=logits.device).view(-1, 1)
            progress_target = (part["window_start"].to(logits.dtype).view(1, -1) + time) / steps.view(1, -1)
        loss, _parts = total_loss(
            logits,
            part["actions"],
            part["loss_mask"],
            aux,
            core.config,
            progress_target=progress_target,
        )
        count = int(part["loss_mask"].sum().item())
        if total_count:
            (loss * (count / total_count)).backward()
        weighted_loss += float(loss.detach()) * count
        supervised = part["loss_mask"].reshape(-1).to(dtype=torch.bool)
        if count:
            chosen = logits.detach().argmax(dim=-1).reshape(-1)
            labels = part["actions"].reshape(-1).long().clamp(min=0)
            correct += int((chosen[supervised] == labels[supervised]).sum().item())
        last_state = step_state.detach()
        del logits, step_state, aux, loss, part
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    core = model.module if hasattr(model, "module") else model
    core.allow_stdp = True
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    mean_loss = weighted_loss / total_count if total_count else 0.0
    parts = {
        "action_ce": mean_loss,
        "loss": mean_loss,
        "supervised_steps": total_count,
        "accuracy": correct / total_count if total_count else 0.0,
    }
    return last_state, parts


def save_checkpoint(path: Path, model: BrainVLNModel, optimizer, step: int, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": model.config.to_dict(),
            "record": record,
        },
        path,
    )


def load_checkpoint(path: Path, model: BrainVLNModel, optimizer=None) -> dict:
    payload = torch.load(path, map_location="cpu")
    incoming = payload["model"]
    current = model.state_dict()
    mismatched = [
        key
        for key, value in current.items()
        if key not in incoming or tuple(incoming[key].shape) != tuple(value.shape)
    ]
    if mismatched:
        raise RuntimeError(
            f"checkpoint is missing {len(mismatched)} tensors, first={mismatched[:5]}"
        )
    model.load_state_dict({key: incoming[key] for key in current}, strict=True)
    if optimizer is not None and payload.get("optimizer") is not None:
        saved = sum(len(group["params"]) for group in payload["optimizer"]["param_groups"])
        live = sum(len(group["params"]) for group in optimizer.state_dict()["param_groups"])
        if saved == live:
            optimizer.load_state_dict(payload["optimizer"])
    return payload
