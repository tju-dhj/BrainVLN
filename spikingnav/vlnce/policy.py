"""Instruction-conditioned SSE + SPN for the four R2R-CE actions."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from spikingnav.config import DEFAULT_CONFIG, SpikingNavConfig
from spikingnav.models.actor_critic import SpikingNavModel
from spikingnav.vlnce.constants import INSTRUCTION_HIDDEN, PAD_INDEX


class InstructionEncoder(nn.Module):
    """GloVe tokens to the 32-d goal vector consumed by SSE."""

    def __init__(
        self,
        embeddings: torch.Tensor,
        goal_dims: int,
        hidden_size: int = INSTRUCTION_HIDDEN,
        pad_index: int = PAD_INDEX,
    ) -> None:
        super().__init__()
        if embeddings.dim() != 2:
            raise ValueError(f"embeddings must be [vocab, dim], got {tuple(embeddings.shape)}")
        self.embed = nn.Embedding.from_pretrained(
            embeddings.float(), freeze=True, padding_idx=pad_index
        )
        self.rnn = nn.LSTM(embeddings.shape[1], hidden_size, batch_first=True)
        self.proj = nn.Linear(hidden_size, goal_dims)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 2:
            raise ValueError(f"tokens must be [N, L], got {tuple(tokens.shape)}")
        lengths = lengths.to(device=tokens.device).long().clamp(min=1, max=tokens.shape[1])
        packed = nn.utils.rnn.pack_padded_sequence(
            self.embed(tokens.long()),
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _outputs, (hidden, _cell) = self.rnn(packed)
        return self.proj(hidden[-1])


class SpikingVLNPolicy(nn.Module):
    """SSE + SPN whose goal input is the instruction encoding."""

    def __init__(
        self,
        embeddings: torch.Tensor,
        cfg: Optional[SpikingNavConfig] = None,
        pretrained_ann: bool = True,
        instruction_hidden: int = INSTRUCTION_HIDDEN,
    ) -> None:
        super().__init__()
        self.cfg = cfg or DEFAULT_CONFIG
        self.encoder = InstructionEncoder(
            embeddings, self.cfg.goal_dims, instruction_hidden
        )
        self.nav = SpikingNavModel(
            action_dim=4,
            num_categories=None,
            cfg=self.cfg,
            pretrained_ann=pretrained_ann,
        )

    def initial_memory(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.nav.initial_memory(batch_size, device)

    def forward(
        self,
        rgb: torch.Tensor,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if rgb.dim() == 4:
            rgb = rgb.unsqueeze(0)
        if rgb.dim() != 5:
            raise ValueError(f"rgb must be [T, N, ...] or [N, ...], got {tuple(rgb.shape)}")
        steps, batch = rgb.shape[:2]
        goal = self.encoder(tokens, lengths)
        if goal.shape[0] != batch:
            raise ValueError(
                f"instruction batch {goal.shape[0]} does not match rgb batch {batch}"
            )
        goal = goal.unsqueeze(0).expand(steps, batch, -1)
        return self.nav(rgb, goal, memory, masks)

    def act(
        self,
        rgb: torch.Tensor,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, _values, memory = self.forward(rgb, tokens, lengths, memory, masks)
        return logits[-1].argmax(dim=-1), memory
