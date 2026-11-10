"""FlyWire comparison helpers. A missing extract is reported as NOT_RUN."""

from __future__ import annotations

import json
from pathlib import Path

import torch

REQUIRED_MANIFEST_FIELDS = ("version", "neuron_ids", "transmitter", "threshold", "license")


def degree_preserving_rewire(pre: torch.Tensor, post: torch.Tensor, seed: int) -> tuple:
    """Permute targets. Each source keeps its out-degree."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    permutation = torch.randperm(post.shape[0], generator=generator)
    return pre.clone(), post[permutation]


def flywire_status(root: Path) -> dict:
    root = Path(root)
    manifest = root / "manifest.json"
    if not manifest.is_file():
        return {
            "status": "NOT_RUN",
            "reason": "no local FlyWire navigation subgraph manifest",
            "expected": str(manifest),
            "required_fields": list(REQUIRED_MANIFEST_FIELDS),
        }
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    missing = [field for field in REQUIRED_MANIFEST_FIELDS if field not in payload]
    if missing:
        return {"status": "NOT_RUN", "reason": f"manifest missing {missing}"}
    return {
        "status": "PRESENT",
        "version": payload["version"],
        "neurons": len(payload["neuron_ids"]),
        "threshold": payload["threshold"],
        "license": payload["license"],
    }
