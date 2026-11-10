"""Masked expert-action loss, with auxiliary terms kept behind their own switches."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_action_ce(logits: torch.Tensor, actions: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """Uniform cross-entropy. Padded labels and burn-in steps contribute nothing."""
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_actions = actions.reshape(-1).long().clamp(min=0)
    flat_mask = loss_mask.reshape(-1).to(logits.dtype)
    per_step = F.cross_entropy(flat_logits, flat_actions, reduction="none")
    return (per_step * flat_mask).sum() / flat_mask.sum().clamp(min=1.0)


def progress_loss(predicted: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    error = (predicted.squeeze(-1) - target).pow(2)
    mask = loss_mask.to(error.dtype)
    return (error * mask).sum() / mask.sum().clamp(min=1.0)


def prediction_loss(predicted: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    error = (predicted - target).pow(2).mean(dim=-1)
    mask = loss_mask.to(error.dtype)
    return (error * mask).sum() / mask.sum().clamp(min=1.0)


def activity_loss(firing_rate: torch.Tensor, target: float, loss_mask: torch.Tensor) -> torch.Tensor:
    error = (firing_rate - target).pow(2)
    mask = loss_mask.to(error.dtype)
    return (error * mask).sum() / mask.sum().clamp(min=1.0)


def total_loss(logits, actions, loss_mask, aux, config, progress_target=None, prediction_target=None):
    loss = masked_action_ce(logits, actions, loss_mask)
    parts = {"action_ce": float(loss.detach())}
    if config.progress_weight > 0 and progress_target is not None:
        term = progress_loss(aux["progress"], progress_target, loss_mask)
        loss = loss + config.progress_weight * term
        parts["progress"] = float(term.detach())
    if config.prediction_weight > 0 and prediction_target is not None:
        term = prediction_loss(aux["surprise"].unsqueeze(-1), prediction_target, loss_mask)
        loss = loss + config.prediction_weight * term
        parts["prediction"] = float(term.detach())
    if config.event_weight > 0:
        term = aux["task_switch"].mean()
        loss = loss + config.event_weight * term
        parts["event"] = float(term.detach())
    if config.activity_weight > 0:
        term = activity_loss(aux["firing_rate"], 0.1, loss_mask)
        loss = loss + config.activity_weight * term
        parts["activity"] = float(term.detach())
    if config.covariance_weight > 0:
        loss = loss + config.covariance_weight * aux["covariance"]
        parts["covariance"] = float(aux["covariance"].detach())
    return loss, parts
