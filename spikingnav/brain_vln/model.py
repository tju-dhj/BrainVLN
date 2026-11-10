"""B0-B5 navigation models. Goal coordinates are not model inputs."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from spikingnav.brain_vln.config import BrainVLNConfig
from spikingnav.brain_vln.frontend import Frontend
from spikingnav.brain_vln.memory import EventMemory
from spikingnav.brain_vln.neurons import lif_step, membrane_decay, spike_function
from spikingnav.brain_vln.plasticity import stdp_trace_step, stdp_update_edges
from spikingnav.brain_vln.state import ReservoirState
from spikingnav.brain_vln.synapses import build_graph, write_spike

VARIANTS = ("B0", "B1", "B2", "B3", "B4", "B5")
MODALITY_WIDTH = 256
SENSORY_SCALE = 0.5
ACTION_SCALE = 0.25
CELL_CHANNELS = 32
RGB_GRID = (3, 3)
DEPTH_GRID = (4, 4)
RGB_CELL_COUNT = RGB_GRID[0] * RGB_GRID[1] * CELL_CHANNELS
DEPTH_CELL_COUNT = DEPTH_GRID[0] * DEPTH_GRID[1] * CELL_CHANNELS
SPATIAL_CELL_COUNT = RGB_CELL_COUNT + DEPTH_CELL_COUNT
CLIP_RGB_CHANNELS = 768
DEPTH_CHANNELS = 192
DIRECT_ENCODERS = ("mlanet", "metaspikeformer")


def modality_current(features: torch.Tensor, scale: float = SENSORY_SCALE) -> torch.Tensor:
    """Center each 256-d half, then apply one fixed gain.

    RGB and depth, or the instruction and its sub-instructions, are normalized
    separately. A joint normalization lets the larger modality set the scale.
    There is no learnable affine. The gain keeps the current near the LIF
    threshold: with leak ``exp(-1/2)`` the resting membrane is about ``2.54``
    times the current, so a standard deviation of ``0.5`` sits near threshold 1.
    """
    width = MODALITY_WIDTH
    if features.shape[-1] != width * 2:
        raise ValueError(f"expected {width * 2} features, got {features.shape[-1]}")
    left = F.layer_norm(features[..., :width], (width,))
    right = F.layer_norm(features[..., width:], (width,))
    return torch.cat((left, right), dim=-1) * scale


def cell_currents(feature_map: torch.Tensor, scale: float = SENSORY_SCALE) -> torch.Tensor:
    """Normalize channels inside each grid cell, then apply the fixed gain.

    ``feature_map`` is ``[B, C, H, W]``. The result is ``[B, H*W*C]`` in
    row, column, channel order, so one index is one cell and one channel.
    Cells do not share a mean or a variance.
    """
    if feature_map.dim() != 4:
        raise ValueError("feature map must be [B, C, H, W]")
    batch, channels, _height, _width = feature_map.shape
    cells = feature_map.permute(0, 2, 3, 1).reshape(batch, -1, channels)
    cells = F.layer_norm(cells, (channels,))
    return (cells * scale).reshape(batch, -1)


class BrainVLNModel(nn.Module):
    def __init__(self, config: Optional[BrainVLNConfig] = None) -> None:
        super().__init__()
        self.config = config or BrainVLNConfig()
        if self.config.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        self.direct_drive = self.config.encoder in DIRECT_ENCODERS
        self.spatial_layout = self.config.visual_layout
        self.v5 = self.config.brain == "v5"
        if self.config.brain not in ("v2", "v5"):
            raise ValueError("brain must be v2 or v5")
        if self.v5:
            if self.config.encoder != "mlanet" or self.spatial_layout != "cells":
                raise ValueError("v5 keeps the frozen MLANet cell layout")
            if self.config.variant != "B4":
                raise ValueError("v5 uses the B4 synaptic graph")
            if self.config.populations.episodic < SPATIAL_CELL_COUNT:
                raise ValueError("v5 episodic population must hold the 800 spatial cells")
            if self.config.stop_neurons < 1 or self.config.episodic_tau <= 0:
                raise ValueError("v5 needs a positive STOP population and episodic tau")
        if self.spatial_layout not in ("pooled", "cells"):
            raise ValueError("visual_layout must be pooled or cells")
        if self.spatial_layout == "cells" and not self.direct_drive:
            raise ValueError("cell layout requires a frozen direct-drive encoder")
        if self.config.encoder == "metaspikeformer" and self.spatial_layout != "cells":
            raise ValueError("metaspikeformer requires visual_layout=cells")
        if self.direct_drive:
            if self.config.variant != "B4":
                raise ValueError("direct current drive is wired for B4")
            sizes = self.config.populations
            if self.spatial_layout == "cells":
                if sizes.spatial != SPATIAL_CELL_COUNT or sizes.goal != MODALITY_WIDTH * 2:
                    raise ValueError("cells layout uses Spatial=800 and Goal=512")
            elif sizes.spatial != MODALITY_WIDTH * 2 or sizes.goal != MODALITY_WIDTH * 2:
                raise ValueError("pooled MLANet drive uses Spatial=512 and Goal=512")
        self.frontend = Frontend(self.config)
        self.separate_inputs = self.config.variant in ("B4", "B5")
        self.use_events = self.config.variant == "B5"
        topology = "random_sparse" if self.config.variant == "B2" else "functionally_structured"
        self.graph = build_graph(self.config, topology) if self.config.variant in ("B2", "B3", "B4", "B5") else None
        width = self.config.populations.total
        feature = self.config.feature_dim
        if self.direct_drive:
            self.mixed_proj = None
            self.spatial_proj = None
            self.goal_proj = None
            self.episodic_proj = None
            self.drive_action = nn.Embedding(5, 32)
            self.action_current = nn.Linear(32, sizes.spatial)
            nn.init.normal_(self.action_current.weight, std=0.02)
            nn.init.zeros_(self.action_current.bias)
            if self.spatial_layout == "cells":
                rgb_channels = CLIP_RGB_CHANNELS
                pretrained = getattr(self.frontend, "pretrained", None)
                if pretrained is not None and hasattr(pretrained, "rgb_channels"):
                    rgb_channels = int(pretrained.rgb_channels)
                self.rgb_cell = nn.Conv2d(rgb_channels, CELL_CHANNELS, kernel_size=1)
                self.depth_cell = nn.Conv2d(DEPTH_CHANNELS, CELL_CHANNELS, kernel_size=1)
            else:
                self.rgb_cell = None
                self.depth_cell = None
        else:
            self.mixed_proj = nn.Linear(feature * 3, width)
            self.spatial_proj = nn.Linear(feature * 2, self.config.populations.spatial)
            self.goal_proj = nn.Linear(feature * 2, self.config.populations.goal)
            self.episodic_proj = nn.Linear(feature, self.config.populations.episodic)
        sizes = self.config.populations
        readout_in = sizes.action + sizes.goal
        if self.v5:
            readout_in += sizes.episodic
        self.readout = nn.Linear(readout_in, 4)
        self.progress_head = nn.Linear(sizes.episodic, 1)
        self.episodic_rest = None
        self.sub_pointer = None
        self.stop_from_episodic = None
        self.stop_from_goal = None
        self.stop_logit = None
        if self.v5:
            rest = sizes.episodic - SPATIAL_CELL_COUNT
            self.episodic_rest = nn.Linear(SPATIAL_CELL_COUNT, rest)
            self.sub_pointer = nn.Linear(sizes.episodic, 12)
            self.stop_from_episodic = nn.Linear(sizes.episodic, self.config.stop_neurons)
            self.stop_from_goal = nn.Linear(sizes.goal, self.config.stop_neurons)
            self.stop_logit = nn.Linear(self.config.stop_neurons, 1)
            nn.init.normal_(self.episodic_rest.weight, std=0.02)
            nn.init.zeros_(self.episodic_rest.bias)
            nn.init.normal_(self.sub_pointer.weight, std=0.02)
            nn.init.zeros_(self.sub_pointer.bias)
            nn.init.normal_(self.stop_from_episodic.weight, std=0.02)
            nn.init.zeros_(self.stop_from_episodic.bias)
            nn.init.normal_(self.stop_from_goal.weight, std=0.02)
            nn.init.zeros_(self.stop_from_goal.bias)
            nn.init.zeros_(self.stop_logit.weight)
            nn.init.zeros_(self.stop_logit.bias)
        self.recurrent = nn.GRUCell(feature * 3, feature)
        self.ann_readout = nn.Linear(feature, 4)
        self.events = (
            EventMemory(
                feature,
                mode=self.config.event_mode,
                threshold=self.config.surprise_threshold,
                period=self.config.event_period,
                max_staleness=self.config.max_staleness,
                skip_inactive=self.config.skip_inactive_memory,
            )
            if self.use_events
            else None
        )
        self.leak = membrane_decay(self.config.dt, self.config.tau_m)
        self.bounds = self.config.populations.bounds()
        self.stop_leak = self.leak
        if self.v5:
            # Two internal steps of this leak equal exp(-1/tau) per environment step.
            # v2 keeps the scalar leak and does not grow its checkpoint.
            leak = torch.full((self.config.populations.total,), self.leak)
            slow = membrane_decay(
                self.config.dt, self.config.episodic_tau * self.config.internal_steps
            )
            start, stop = self.bounds["episodic"]
            leak[start:stop] = slow
            self.register_buffer("neuron_leak", leak)
            self.stop_leak = membrane_decay(self.config.dt, self.config.episodic_tau)
        self.use_stdp = bool(self.config.use_stdp)
        if self.use_stdp and self.graph is None:
            raise ValueError("STDP requires a synaptic graph (variants B2-B5)")
        # Gate online STDP so gradient-checkpoint recomputes do not double-apply it.
        self.allow_stdp = True
        if self.spatial_layout == "cells":
            self._trace_dim = SPATIAL_CELL_COUNT
        elif self.direct_drive:
            self._trace_dim = MODALITY_WIDTH * 2
        else:
            self._trace_dim = self.config.feature_dim

    def initial_state(self, batch: int, device: torch.device) -> ReservoirState:
        width = self.config.populations.total
        edges = 0 if self.graph is None else self.graph.n_edges
        span = 1 if self.graph is None else self.graph.max_delay + 1
        stdp_width = width if self.use_stdp else 0
        return ReservoirState(
            membrane=torch.zeros(batch, width, device=device),
            spike=torch.zeros(batch, width, device=device),
            adaptation=torch.zeros(batch, width, device=device),
            synaptic_current=torch.zeros(batch, edges, device=device),
            delay_queue=torch.zeros(batch, span, width, device=device),
            time=torch.zeros(batch, dtype=torch.long, device=device),
            spatial_trace=torch.zeros(batch, self._trace_dim, device=device),
            staleness=torch.zeros(batch, device=device),
            hidden=torch.zeros(batch, self.config.feature_dim, device=device),
            pre_trace=torch.zeros(batch, stdp_width, device=device),
            post_trace=torch.zeros(batch, stdp_width, device=device),
            stop_membrane=torch.zeros(batch, self.config.stop_neurons if self.v5 else 0, device=device),
        )

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        depth_valid: torch.Tensor,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        valid_mask: torch.Tensor,
        reset_mask: torch.Tensor,
        detach_mask: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        prev_action: Optional[torch.Tensor] = None,
        state: Optional[ReservoirState] = None,
        fsa_state: Optional[torch.Tensor] = None,
        spans: Optional[torch.Tensor] = None,
        instruction_text=None,
    ):
        if rgb.dim() != 5:
            raise ValueError("rgb must be [T, B, 3, H, W]")
        steps, batch = rgb.shape[:2]
        if state is None:
            state = self.initial_state(batch, rgb.device)
        self.frontend.prepare_language(tokens, token_mask, instruction_text)
        outputs = []
        surprises = []
        switches = []
        rates = []
        progress = []
        previous_fsa = None
        for step in range(steps):
            reset = reset_mask[step]
            if bool(reset.any()):
                state = state.zero_rows(reset)
                previous_fsa = None
            if bool(detach_mask[step].any()):
                state = state.detach()
            if step == 0 and prev_action is not None:
                history = prev_action.to(device=rgb.device).long().view(batch)
            else:
                history = _previous_action(actions, step, batch, rgb.device)
            fsa_now = None if fsa_state is None else fsa_state[step]
            if torch.is_grad_enabled():
                # The 64-step reservoir graph does not fit a batch of 128 on a
                # 32GB GPU. Recompute one step on the backward pass. The loss
                # and the parameter gradients stay the same.
                packed = checkpoint(
                    self._checkpoint_step,
                    rgb[step],
                    depth[step],
                    depth_valid[step],
                    history,
                    state.membrane,
                    state.spike,
                    state.adaptation,
                    state.synaptic_current,
                    state.delay_queue,
                    state.time,
                    state.spatial_trace,
                    state.staleness,
                    state.hidden,
                    state.pre_trace,
                    state.post_trace,
                    state.stop_membrane,
                    valid_mask[step],
                    reset,
                    tokens,
                    token_mask,
                    use_reentrant=False,
                    fsa_now=fsa_now,
                    previous_fsa=previous_fsa,
                    spans=spans,
                )
                logits = packed[0]
                info = _info(packed[1], packed[2], packed[3], packed[4])
                state = ReservoirState(
                    membrane=packed[5],
                    spike=packed[6],
                    adaptation=packed[7],
                    synaptic_current=packed[8],
                    delay_queue=packed[9],
                    time=packed[10],
                    spatial_trace=packed[11],
                    staleness=packed[12],
                    hidden=packed[13],
                    pre_trace=packed[14],
                    post_trace=packed[15],
                    stop_membrane=packed[16],
                )
            else:
                logits, state, info = self._step(
                    rgb[step],
                    depth[step],
                    depth_valid[step],
                    tokens,
                    token_mask,
                    history,
                    state,
                    valid_mask[step],
                    reset,
                    fsa_now,
                    previous_fsa,
                    spans,
                )
            previous_fsa = None if fsa_now is None else fsa_now.detach()
            outputs.append(logits)
            surprises.append(info["surprise"])
            switches.append(info["task_switch"])
            rates.append(info["firing_rate"])
            progress.append(info["progress"])
        aux = {
            "surprise": torch.stack(surprises, dim=0),
            "task_switch": torch.stack(switches, dim=0),
            "firing_rate": torch.stack(rates, dim=0),
            "progress": torch.stack(progress, dim=0),
            "covariance": state.membrane.new_zeros(()),
        }
        if self.separate_inputs:
            aux["covariance"] = _covariance(state.membrane, self.bounds)
        return torch.stack(outputs, dim=0), state, aux

    def _checkpoint_step(
        self,
        rgb,
        depth,
        depth_valid,
        prev_action,
        membrane,
        spike,
        adaptation,
        synaptic_current,
        delay_queue,
        time,
        spatial_trace,
        staleness,
        hidden,
        pre_trace,
        post_trace,
        stop_membrane,
        valid,
        reset,
        tokens,
        token_mask,
        fsa_now=None,
        previous_fsa=None,
        spans=None,
    ):
        state = ReservoirState(
            membrane=membrane,
            spike=spike,
            adaptation=adaptation,
            synaptic_current=synaptic_current,
            delay_queue=delay_queue,
            time=time,
            spatial_trace=spatial_trace,
            staleness=staleness,
            hidden=hidden,
            pre_trace=pre_trace,
            post_trace=post_trace,
            stop_membrane=stop_membrane,
        )
        logits, new_state, info = self._step(
            rgb,
            depth,
            depth_valid,
            tokens,
            token_mask,
            prev_action,
            state,
            valid,
            reset,
            fsa_now,
            previous_fsa,
            spans,
        )
        return (
            logits,
            info["surprise"],
            info["task_switch"],
            info["firing_rate"],
            info["progress"],
            new_state.membrane,
            new_state.spike,
            new_state.adaptation,
            new_state.synaptic_current,
            new_state.delay_queue,
            new_state.time,
            new_state.spatial_trace,
            new_state.staleness,
            new_state.hidden,
            new_state.pre_trace,
            new_state.post_trace,
            new_state.stop_membrane,
        )

    def _step(
        self,
        rgb,
        depth,
        depth_valid,
        tokens,
        token_mask,
        prev_action,
        state: ReservoirState,
        valid: torch.Tensor,
        reset: torch.Tensor,
        fsa_now,
        previous_fsa,
        spans,
    ):
        encoded = self.frontend(
            rgb, depth, depth_valid, tokens, token_mask, prev_action, spans, reset=reset
        )
        return self._step_encoded(
            encoded, state, valid, reset, fsa_now=fsa_now, previous_fsa=previous_fsa
        )

    def forward_from_maps(
        self,
        rgb_map: torch.Tensor,
        depth_map: torch.Tensor,
        language: torch.Tensor,
        prev_actions: torch.Tensor,
        valid_mask: torch.Tensor,
        reset_mask: torch.Tensor,
        detach_mask: torch.Tensor,
        state: Optional[ReservoirState] = None,
    ):
        """Reservoir step on frozen MLANet maps.

        ``rgb_map`` is ``[T, B, 768, 3, 3]``, ``depth_map`` is ``[T, B, 192, 4, 4]``,
        and ``language`` is the 512-d instruction current ``[B, 512]``. Previous
        actions are the actions actually executed, including the start id 4.
        """
        if self.v5:
            raise ValueError(
                "v5 selects a sub-instruction from the language bank, which map batches do not carry"
            )
        if self.rgb_cell is None or self.depth_cell is None:
            raise ValueError("map drive requires the cells layout")
        if rgb_map.dim() != 5 or depth_map.dim() != 5:
            raise ValueError("maps must be [T, B, C, H, W]")
        steps, batch = rgb_map.shape[:2]
        if state is None:
            state = self.initial_state(batch, rgb_map.device)
        outputs = []
        surprises = []
        switches = []
        rates = []
        progress = []
        for step in range(steps):
            reset = reset_mask[step]
            if bool(reset.any()):
                state = state.zero_rows(reset)
            if bool(detach_mask[step].any()):
                state = state.detach()
            if torch.is_grad_enabled():
                packed = checkpoint(
                    self._checkpoint_from_maps,
                    rgb_map[step],
                    depth_map[step],
                    language,
                    prev_actions[step],
                    state.membrane,
                    state.spike,
                    state.adaptation,
                    state.synaptic_current,
                    state.delay_queue,
                    state.time,
                    state.spatial_trace,
                    state.staleness,
                    state.hidden,
                    state.pre_trace,
                    state.post_trace,
                    state.stop_membrane,
                    valid_mask[step],
                    reset,
                    use_reentrant=False,
                )
                logits = packed[0]
                info = _info(packed[1], packed[2], packed[3], packed[4])
                state = ReservoirState(
                    membrane=packed[5],
                    spike=packed[6],
                    adaptation=packed[7],
                    synaptic_current=packed[8],
                    delay_queue=packed[9],
                    time=packed[10],
                    spatial_trace=packed[11],
                    staleness=packed[12],
                    hidden=packed[13],
                    pre_trace=packed[14],
                    post_trace=packed[15],
                    stop_membrane=packed[16],
                )
            else:
                logits, state, info = self._step_encoded(
                    self._encoded_from_maps(rgb_map[step], depth_map[step], language, prev_actions[step]),
                    state,
                    valid_mask[step],
                    reset,
                )
            outputs.append(logits)
            surprises.append(info["surprise"])
            switches.append(info["task_switch"])
            rates.append(info["firing_rate"])
            progress.append(info["progress"])
        aux = {
            "surprise": torch.stack(surprises, dim=0),
            "task_switch": torch.stack(switches, dim=0),
            "firing_rate": torch.stack(rates, dim=0),
            "progress": torch.stack(progress, dim=0),
            "covariance": state.membrane.new_zeros(()),
        }
        if self.separate_inputs:
            aux["covariance"] = _covariance(state.membrane, self.bounds)
        return torch.stack(outputs, dim=0), state, aux

    def _encoded_from_maps(self, rgb_map, depth_map, language, prev_action) -> dict:
        batch = rgb_map.shape[0]
        return {
            "rgb": language,
            "depth": language,
            "fused": language,
            "instruction": language,
            "subinstruction": language.new_zeros(batch, self.config.feature_dim),
            "subinstruction_present": language.new_zeros(batch),
            "history": language.new_zeros(batch, self.config.feature_dim),
            "prev_action": prev_action.long().view(-1).clamp(0, 4),
            "rgb_map": rgb_map,
            "depth_map": depth_map,
        }

    def _checkpoint_from_maps(
        self,
        rgb_map,
        depth_map,
        language,
        prev_action,
        membrane,
        spike,
        adaptation,
        synaptic_current,
        delay_queue,
        time,
        spatial_trace,
        staleness,
        hidden,
        pre_trace,
        post_trace,
        stop_membrane,
        valid,
        reset,
    ):
        state = ReservoirState(
            membrane=membrane,
            spike=spike,
            adaptation=adaptation,
            synaptic_current=synaptic_current,
            delay_queue=delay_queue,
            time=time,
            spatial_trace=spatial_trace,
            staleness=staleness,
            hidden=hidden,
            pre_trace=pre_trace,
            post_trace=post_trace,
            stop_membrane=stop_membrane,
        )
        logits, new_state, info = self._step_encoded(
            self._encoded_from_maps(rgb_map, depth_map, language, prev_action),
            state,
            valid,
            reset,
        )
        return (
            logits,
            info["surprise"],
            info["task_switch"],
            info["firing_rate"],
            info["progress"],
            new_state.membrane,
            new_state.spike,
            new_state.adaptation,
            new_state.synaptic_current,
            new_state.delay_queue,
            new_state.time,
            new_state.spatial_trace,
            new_state.staleness,
            new_state.hidden,
            new_state.pre_trace,
            new_state.post_trace,
            new_state.stop_membrane,
        )

    def _step_encoded(
        self,
        encoded,
        state: ReservoirState,
        valid: torch.Tensor,
        reset: torch.Tensor,
        fsa_now=None,
        previous_fsa=None,
    ):
        reference = encoded["fused"]
        batch = reference.shape[0]
        logits = reference.new_zeros(batch, 4)
        surprise = reference.new_zeros(batch)
        task_switch = reference.new_zeros(batch)
        progress = reference.new_zeros(batch, 1)
        rate = reference.new_zeros(batch)
        if not bool(valid.any()):
            return logits, state, _info(surprise, task_switch, rate, progress)
        if self.config.variant == "B0":
            features = torch.cat(
                (encoded["fused"], encoded["instruction"], encoded["history"]), dim=-1
            )
            hidden = self.recurrent(features, state.hidden)
            step_logits = self.ann_readout(hidden)
            new_state = ReservoirState(
                membrane=state.membrane,
                spike=state.spike,
                adaptation=state.adaptation,
                synaptic_current=state.synaptic_current,
                delay_queue=state.delay_queue,
                time=state.time,
                spatial_trace=encoded["fused"].detach(),
                staleness=state.staleness,
                hidden=hidden,
                pre_trace=state.pre_trace,
                post_trace=state.post_trace,
                stop_membrane=state.stop_membrane,
            )
            return _mask_step(step_logits, state, new_state, valid, surprise, task_switch, rate, progress)

        external, surprise, task_switch, staleness, trace = self._external(encoded, state, fsa_now, previous_fsa)
        membrane = state.membrane
        spike = state.spike
        adaptation = state.adaptation
        current = state.synaptic_current
        queue = state.delay_queue
        clock = state.time
        pre_trace = state.pre_trace
        post_trace = state.post_trace
        increment = 0.0 if self.config.variant == "B1" else self.config.adaptation_increment
        apply_stdp = (
            self.use_stdp
            and self.training
            and self.allow_stdp
            and self.graph is not None
        )
        for _ in range(self.config.internal_steps):
            if self.graph is None:
                post = torch.zeros_like(membrane)
                current = state.synaptic_current
            else:
                current, post = self.graph.step(queue, clock, current)
            membrane, spike, adaptation = lif_step(
                membrane,
                post + external,
                adaptation,
                leak=self.neuron_leak if self.v5 else self.leak,
                threshold=self.config.threshold,
                surrogate=self.config.surrogate,
                surrogate_alpha=self.config.surrogate_alpha,
                adaptation_decay=self.config.adaptation_decay,
                adaptation_increment=increment,
            )
            if apply_stdp:
                hard_spike = spike.detach()
                pre_trace, post_trace = stdp_trace_step(
                    hard_spike,
                    pre_trace,
                    post_trace,
                    tau=self.config.stdp_tau,
                    dt=self.config.dt,
                )
                with torch.no_grad():
                    updated = stdp_update_edges(
                        self.graph.weight.data,
                        self.graph.pre,
                        self.graph.post,
                        self.graph.sign,
                        hard_spike,
                        pre_trace,
                        post_trace,
                        a_pos=self.config.stdp_a_pos,
                        a_neg=self.config.stdp_a_neg,
                        excitatory_only=self.config.stdp_excitatory_only,
                    )
                    self.graph.weight.data.copy_(updated)
            queue = write_spike(queue, clock, spike)
            clock = clock + 1
        goal = _slice(membrane, self.bounds["goal"])
        action = _slice(membrane, self.bounds["action"])
        episodic = _slice(membrane, self.bounds["episodic"])
        if self.v5:
            step_logits = self.readout(torch.cat((action, goal, episodic), dim=-1))
            drive = self.stop_from_episodic(episodic) + self.stop_from_goal(goal)
            pre_stop = self.stop_leak * state.stop_membrane + drive
            stop_spike = spike_function(
                pre_stop - self.config.threshold,
                self.config.surrogate,
                self.config.surrogate_alpha,
            )
            stop_membrane = pre_stop * (1.0 - stop_spike)
            step_logits = step_logits.clone()
            step_logits[:, 0] = step_logits[:, 0] + self.stop_logit(stop_membrane).squeeze(-1)
        else:
            step_logits = self.readout(torch.cat((action, goal), dim=-1))
            stop_membrane = state.stop_membrane
        progress = self.progress_head(episodic)
        rate = spike.mean(dim=-1)
        new_state = ReservoirState(
            membrane=membrane,
            spike=spike,
            adaptation=adaptation,
            synaptic_current=current,
            delay_queue=queue,
            time=clock,
            spatial_trace=trace,
            staleness=staleness,
            hidden=state.hidden,
            pre_trace=pre_trace,
            post_trace=post_trace,
            stop_membrane=stop_membrane,
        )
        return _mask_step(step_logits, state, new_state, valid, surprise, task_switch, rate, progress)

    def _external(self, encoded, state: ReservoirState, fsa_now, previous_fsa):
        batch = encoded["fused"].shape[0]
        external = encoded["fused"].new_zeros(batch, self.config.populations.total)
        surprise = encoded["fused"].new_zeros(batch)
        task_switch = encoded["fused"].new_zeros(batch)
        staleness = state.staleness
        trace = encoded["fused"]
        if self.direct_drive:
            if self.spatial_layout == "cells":
                rgb_cells = cell_currents(self.rgb_cell(encoded["rgb_map"]))
                depth_cells = cell_currents(self.depth_cell(encoded["depth_map"]))
                spatial = torch.cat((rgb_cells, depth_cells), dim=-1)
            else:
                spatial = modality_current(encoded["fused"], SENSORY_SCALE)
            action_code = self.drive_action(encoded["prev_action"].long().view(-1).clamp(0, 4))
            view = spatial
            spatial = view + ACTION_SCALE * torch.tanh(self.action_current(action_code))
            if self.v5:
                goal = self._goal_current(encoded, state)
            else:
                goal = modality_current(encoded["instruction"], SENSORY_SCALE)
            external[:, self.bounds["spatial"][0] : self.bounds["spatial"][1]] = spatial
            external[:, self.bounds["goal"][0] : self.bounds["goal"][1]] = goal
            trace = view if self.v5 else spatial
            if self.v5:
                # A closed gate leaves the episodic current at zero, so an
                # unchanged view is kept by the slow leak instead of rewritten.
                surprise = (view - state.spatial_trace).pow(2).mean(dim=-1)
                gate = (surprise > self.config.view_change_threshold).to(dtype=view.dtype).unsqueeze(-1)
                start, stop = self.bounds["episodic"]
                external[:, start:stop] = torch.cat(
                    (gate * view, gate * self.episodic_rest(view)), dim=-1
                )
        elif self.separate_inputs:
            external[:, self.bounds["spatial"][0] : self.bounds["spatial"][1]] = self.spatial_proj(
                torch.cat((encoded["fused"], encoded["history"]), dim=-1)
            )
            external[:, self.bounds["goal"][0] : self.bounds["goal"][1]] = self.goal_proj(
                torch.cat((encoded["instruction"], encoded["subinstruction"]), dim=-1)
            )
        else:
            external = self.mixed_proj(
                torch.cat((encoded["fused"], encoded["instruction"], encoded["history"]), dim=-1)
            )
        if self.events is not None:
            episodic = _slice(state.membrane, self.bounds["episodic"]).mean(dim=-1, keepdim=True)
            episodic = episodic.expand(-1, self.config.feature_dim)
            event = self.events(
                encoded["fused"],
                state.spatial_trace,
                encoded["instruction"],
                episodic,
                encoded["fused"],
                state.staleness,
                state.time,
                fsa_now,
                previous_fsa,
            )
            external[:, self.bounds["episodic"][0] : self.bounds["episodic"][1]] = self.episodic_proj(
                event["write"]
            )
            surprise = event["surprise"]
            task_switch = event["task_switch"]
            staleness = event["staleness"]
        return external, surprise, task_switch, staleness, trace.detach()

    def _goal_current(self, encoded, state: ReservoirState) -> torch.Tensor:
        sentence = encoded["sentence"]
        subs = encoded["sub_bank"]
        if sentence.shape[-1] != MODALITY_WIDTH or subs.shape[-1] != MODALITY_WIDTH or subs.shape[1] != 12:
            raise ValueError(
                f"v5 expects a 256-d sentence and 12x256 sub-instructions, got "
                f"{tuple(sentence.shape)} and {tuple(subs.shape)}"
            )
        episodic = _slice(state.membrane, self.bounds["episodic"])
        mask = encoded["sub_mask"].to(device=episodic.device, dtype=torch.bool)
        if mask.shape != (sentence.shape[0], 12):
            raise ValueError(f"v5 sub-instruction mask must be [batch, 12], got {tuple(mask.shape)}")
        scores = self.sub_pointer(episodic)
        usable = ~mask
        # Padding is removed before the softmax. A row with no real clause
        # would otherwise be all minima and the softmax would be NaN.
        none = ~usable.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(~usable, torch.finfo(scores.dtype).min)
        scores = torch.where(none, torch.zeros_like(scores), scores)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(none, torch.zeros_like(weights), weights)
        selected = torch.einsum("bs,bsd->bd", weights, subs)
        return modality_current(torch.cat((sentence, selected), dim=-1))


def _previous_action(actions, step: int, batch: int, device) -> torch.Tensor:
    if actions is None or step == 0:
        return torch.full((batch,), 4, dtype=torch.long, device=device)
    return actions[step - 1].clamp(min=0, max=3).long()


def _slice(tensor: torch.Tensor, bounds) -> torch.Tensor:
    start, stop = bounds
    return tensor[:, start:stop]


def _covariance(membrane: torch.Tensor, bounds) -> torch.Tensor:
    spatial = _slice(membrane, bounds["spatial"]).mean(dim=-1)
    goal = _slice(membrane, bounds["goal"]).mean(dim=-1)
    if spatial.shape[0] < 2:
        return membrane.new_zeros(())
    spatial = spatial - spatial.mean()
    goal = goal - goal.mean()
    return (spatial * goal).mean().pow(2)


def _info(surprise, task_switch, rate, progress):
    return {
        "surprise": surprise,
        "task_switch": task_switch,
        "firing_rate": rate,
        "progress": progress,
    }


def _mask_step(logits, old: ReservoirState, new: ReservoirState, valid, surprise, task_switch, rate, progress):
    mask = valid.view(-1, *([1] * (logits.dim() - 1)))
    logits = torch.where(mask, logits, torch.zeros_like(logits))
    surprise = torch.where(valid, surprise, torch.zeros_like(surprise))
    task_switch = torch.where(valid, task_switch, torch.zeros_like(task_switch))
    rate = torch.where(valid, rate, torch.zeros_like(rate))
    progress = torch.where(valid.view(-1, 1), progress, torch.zeros_like(progress))
    mixed = ReservoirState(
        membrane=_where_rows(new.membrane, old.membrane, valid),
        spike=_where_rows(new.spike, old.spike, valid),
        adaptation=_where_rows(new.adaptation, old.adaptation, valid),
        synaptic_current=_where_rows(new.synaptic_current, old.synaptic_current, valid),
        delay_queue=_where_rows(new.delay_queue, old.delay_queue, valid),
        time=torch.where(valid, new.time, old.time),
        spatial_trace=_where_rows(new.spatial_trace, old.spatial_trace, valid),
        staleness=torch.where(valid, new.staleness, old.staleness),
        hidden=_where_rows(new.hidden, old.hidden, valid),
        pre_trace=_where_rows(new.pre_trace, old.pre_trace, valid),
        post_trace=_where_rows(new.post_trace, old.post_trace, valid),
        stop_membrane=_where_rows(new.stop_membrane, old.stop_membrane, valid),
    )
    return logits, mixed, _info(surprise, task_switch, rate, progress)


def _where_rows(updated, previous, valid: torch.Tensor):
    view = valid.view(valid.shape[0], *([1] * (updated.dim() - 1)))
    return torch.where(view, updated, previous)
