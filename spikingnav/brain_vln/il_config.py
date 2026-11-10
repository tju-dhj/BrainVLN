"""Turn configs/brain_vln_il.yaml into the settings the imitation trainer uses."""

from __future__ import annotations

from types import SimpleNamespace

from omegaconf import OmegaConf

_TOP = {
    "output",
    "root",
    "device",
    "seed",
    "brain",
    "variant",
    "layout",
    "encoder",
    "stdp",
    "epochs",
    "batch_size",
    "microbatch",
    "workers",
    "lr",
    "grad_clip",
    "checkpoint_every",
    "data_parallel",
    "resume",
    "episodic_tau",
    "stop_neurons",
    "view_change_threshold",
    "stdp_a_pos",
    "stdp_a_neg",
    "stdp_tau",
}
_VARIANTS = {"B0", "B1", "B2", "B3", "B4", "B5"}
_LAYOUTS = {"pooled", "cells"}
_ENCODERS = {"mlanet", "metaspikeformer"}
_BRAINS = {"v2", "v5"}


def settings_from_config(cfg) -> SimpleNamespace:
    """Validate a Hydra config. Unknown keys are rejected."""
    values = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(values, dict):
        raise TypeError("IL config must be a mapping")
    values.pop("hydra", None)
    _exact_keys(values, _TOP, "config")
    for key in ("output", "root", "device"):
        if not str(values[key]).strip():
            raise ValueError(f"{key} must be set in configs/brain_vln_il.yaml")
    for key in ("epochs", "batch_size", "microbatch", "stop_neurons"):
        if int(values[key]) < 1:
            raise ValueError(f"{key} must be positive")
    if int(values["workers"]) < 0 or int(values["checkpoint_every"]) < 0:
        raise ValueError("workers and checkpoint_every must be zero or positive")
    if float(values["lr"]) <= 0 or float(values["episodic_tau"]) <= 0 or float(values["stdp_tau"]) <= 0:
        raise ValueError("lr, episodic_tau, and stdp_tau must be positive")
    if float(values["grad_clip"]) < 0 or float(values["view_change_threshold"]) < 0:
        raise ValueError("grad_clip and view_change_threshold must be zero or positive")
    brain = str(values["brain"])
    variant = str(values["variant"])
    layout = str(values["layout"])
    encoder = str(values["encoder"])
    stdp = bool(values["stdp"])
    if brain not in _BRAINS:
        raise ValueError("brain must be v2 or v5")
    if variant not in _VARIANTS:
        raise ValueError(f"variant must be one of {sorted(_VARIANTS)}")
    if layout not in _LAYOUTS:
        raise ValueError("layout must be pooled or cells")
    if encoder not in _ENCODERS:
        raise ValueError("encoder must be mlanet or metaspikeformer")
    if brain == "v5" and (layout != "cells" or encoder != "mlanet" or stdp or variant != "B4"):
        raise ValueError("BrainVLN_v5 is the MLANet cells B4 reservoir without STDP")
    if stdp and layout != "cells":
        raise ValueError("BrainVLN_v3 STDP training uses the cells layout")
    if stdp and encoder != "mlanet":
        raise ValueError("BrainVLN_v3 STDP training keeps the MLANet encoder")
    if encoder == "metaspikeformer" and layout != "cells":
        raise ValueError("BrainVLN_v4 uses the cells layout")
    if encoder == "metaspikeformer" and stdp:
        raise ValueError("BrainVLN_v4 does not enable STDP")
    resume = str(values["resume"]).strip()
    return SimpleNamespace(
        output=str(values["output"]),
        root=str(values["root"]),
        device=str(values["device"]),
        seed=int(values["seed"]),
        brain=brain,
        variant=variant,
        layout=layout,
        encoder=encoder,
        stdp=stdp,
        epochs=int(values["epochs"]),
        batch_size=int(values["batch_size"]),
        microbatch=int(values["microbatch"]),
        workers=int(values["workers"]),
        lr=float(values["lr"]),
        grad_clip=float(values["grad_clip"]),
        checkpoint_every=int(values["checkpoint_every"]),
        data_parallel=bool(values["data_parallel"]),
        resume=resume,
        episodic_tau=float(values["episodic_tau"]),
        stop_neurons=int(values["stop_neurons"]),
        view_change_threshold=float(values["view_change_threshold"]),
        stdp_a_pos=float(values["stdp_a_pos"]),
        stdp_a_neg=float(values["stdp_a_neg"]),
        stdp_tau=float(values["stdp_tau"]),
    )


def _exact_keys(values: dict, expected: set, label: str) -> None:
    if not isinstance(values, dict):
        raise TypeError(f"{label} must be a mapping")
    missing = sorted(expected - set(values))
    unknown = sorted(set(values) - expected)
    if missing or unknown:
        parts = []
        if missing:
            parts.append("missing " + ", ".join(missing))
        if unknown:
            parts.append("unknown " + ", ".join(unknown))
        raise ValueError(f"{label}: " + "; ".join(parts))
