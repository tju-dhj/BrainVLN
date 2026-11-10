"""Defaults for the synaptic BrainVLN stack.

4096 neurons is the engineering starting point from the execution manual.
Tests and smokes use smaller populations through this same config.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Tuple


@dataclass
class PopulationSizes:
    spatial: int = 1536
    goal: int = 768
    episodic: int = 1280
    action: int = 512

    def __post_init__(self) -> None:
        if min(self.spatial, self.goal, self.episodic, self.action) < 1:
            raise ValueError("every population needs at least one neuron")

    @property
    def total(self) -> int:
        return self.spatial + self.goal + self.episodic + self.action

    def bounds(self) -> Dict[str, Tuple[int, int]]:
        cursor = 0
        ranges = {}
        for name in ("spatial", "goal", "episodic", "action"):
            width = getattr(self, name)
            ranges[name] = (cursor, cursor + width)
            cursor += width
        return ranges


@dataclass
class BrainVLNConfig:
    populations: PopulationSizes = field(default_factory=PopulationSizes)
    variant: str = "B5"
    out_degree: int = 48
    excitatory_fraction: float = 0.8
    delay_min: int = 1
    delay_max: int = 4
    dt: float = 1.0
    tau_m: float = 2.0
    tau_syn: float = 2.0
    threshold: float = 1.0
    adaptation_increment: float = 0.1
    adaptation_decay: float = 0.9
    surrogate: str = "atan"
    surrogate_alpha: float = 2.0
    internal_steps: int = 2
    weight_scale: float = 0.3
    feature_dim: int = 64
    encoder: str = "conv"
    encoder_root: str = "data/pre_trained_model"
    # pooled: one 512-d visual vector. cells: RGB 3x3 and depth 4x4 stay on the grid.
    visual_layout: str = "pooled"
    vocab_size: int = 8192
    pad_index: int = 0
    seed: int = 0
    covariance_weight: float = 0.0
    event_mode: str = "surprise"
    surprise_threshold: float = 0.5
    event_period: int = 4
    max_staleness: int = 8
    cross_attention: bool = False
    trainable_synapses: bool = False
    # BrainVLN_v3: NeuCube-style online STDP on excitatory synaptic magnitudes.
    use_stdp: bool = False
    stdp_a_pos: float = 1e-4
    stdp_a_neg: float = 1e-2
    stdp_tau: float = 3.0
    stdp_excitatory_only: bool = True
    # v2 keeps the B4 cells reservoir. v5 adds a slow episodic write, a
    # sub-instruction pointer, and a STOP population outside the synaptic graph.
    brain: str = "v2"
    episodic_tau: float = 30.0
    stop_neurons: int = 64
    view_change_threshold: float = 0.05
    skip_inactive_memory: bool = True
    progress_weight: float = 0.0
    prediction_weight: float = 0.0
    event_weight: float = 0.0
    activity_weight: float = 0.0
    weight_source: str = (
        "random initialization; the external MLANet PPO checkpoint was not loaded"
    )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["neuron_count"] = self.populations.total
        return payload


def tiny_config(**overrides) -> BrainVLNConfig:
    """Small deterministic config for tests and short smokes."""
    config = BrainVLNConfig(
        populations=PopulationSizes(spatial=8, goal=4, episodic=6, action=4),
        out_degree=2,
        feature_dim=16,
        vocab_size=64,
        internal_steps=2,
        seed=0,
    )
    for key, value in overrides.items():
        if key in ("spatial", "goal", "episodic", "action"):
            raise ValueError("set populations=PopulationSizes(...) instead of individual sizes")
        setattr(config, key, value)
    return config
