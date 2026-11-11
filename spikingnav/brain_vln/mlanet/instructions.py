"""R2R token rows for the frozen MLANet language tower.

Low-level ids are the stored R2R tokens, clipped into the 2504-word
embedding. CLIP sub-instruction rows come from ``R2R_VLNCE_FSASub``.
Episode ids collide across splits, so the lookup key is the instruction text.
Both datasets are read from this repository's ``data/datasets`` or
``storage/datasets``.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

INSTRUCTION_LEN = 100
SUB_NUM = 12
SUB_LEN = 77
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SPLITS = ("train", "val_seen", "val_unseen")
_EMPTY = object()
_CONFLICT = object()


def r2r_tokens(raw, vocab_size: int) -> list:
    tokens = [int(token) if int(token) < vocab_size else 1 for token in (raw or [])]
    if not tokens:
        tokens = [1]
    if len(tokens) > INSTRUCTION_LEN:
        tokens = tokens[:INSTRUCTION_LEN]
    return tokens + [0] * (INSTRUCTION_LEN - len(tokens))


def fit_sub_rows(rows) -> list:
    fitted = []
    for row in list(rows or [])[:SUB_NUM]:
        values = [int(token) for token in row][:SUB_LEN]
        fitted.append(values + [0] * (SUB_LEN - len(values)))
    while len(fitted) < SUB_NUM:
        fitted.append([0] * SUB_LEN)
    return fitted


def text_key(text: str) -> str:
    return " ".join((text or "").split())


def punct_key(text: str) -> str:
    """Drop spacing and repeated-period differences between the two releases.

    ``R2R_VLNCE_v1-3_preprocessed`` sometimes omits the space after ``.``
    and writes ``..`` where ``R2R_VLNCE_FSASub`` has one period. FSASub's
    ``*_sub`` files also drop a leading ``. ``. The words are the same instruction.
    """
    collapsed = " ".join((text or "").split())
    collapsed = re.sub(r"\.+", ".", collapsed)
    collapsed = collapsed.lstrip(". ")
    return re.sub(r"([.!?])\s+", r"\1", collapsed)


class OfficialInstructions:
    def __init__(
        self,
        r2r: dict,
        fsa_tokens: dict,
        fsa_subs: dict,
        fsa_tokens_punct: dict,
        fsa_subs_punct: dict,
    ) -> None:
        self.r2r = r2r
        self.fsa_tokens = fsa_tokens
        self.fsa_subs = fsa_subs
        self.fsa_tokens_punct = fsa_tokens_punct
        self.fsa_subs_punct = fsa_subs_punct

    def mlanet_tokens(self, text: str):
        found = self.fsa_tokens.get(text_key(text))
        if found is not None:
            return found
        found = self.fsa_tokens_punct.get(punct_key(text))
        if found is not None:
            return found
        return self.r2r.get(text_key(text))

    def sub_tokens(self, text: str):
        found = self.fsa_subs.get(text_key(text))
        if found is not None:
            return found
        return self.fsa_subs_punct.get(punct_key(text))


def _dataset_dir(name: str) -> Path:
    tried = []
    for base in (
        _PROJECT_ROOT / "data" / "datasets",
        _PROJECT_ROOT / "storage" / "datasets",
    ):
        for path in (base / name / name, base / name):
            tried.append(path)
            if (path / "train").is_dir():
                return path
    looked = "\n".join(str(path) for path in tried)
    raise FileNotFoundError(f"put {name} inside this project. Looked for a train split under:\n{looked}")


def _read_json(path: Path):
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def _text(episode: dict) -> str:
    instruction = episode.get("instruction") or {}
    if isinstance(instruction, dict):
        return instruction.get("instruction_text") or ""
    return ""


def _tokens(episode: dict) -> list:
    instruction = episode.get("instruction") or {}
    raw = instruction.get("instruction_tokens") if isinstance(instruction, dict) else None
    tokens = [int(token) for token in (raw or [])]
    while tokens and tokens[-1] == 0:
        tokens.pop()
    return tokens


def _remember_punct(store: dict, key: str, value) -> None:
    current = store.get(key, _EMPTY)
    if current is _EMPTY:
        store[key] = value
    elif current != value:
        store[key] = _CONFLICT


def _index_split(
    path: Path,
    tokens_out: dict,
    subs_out: dict | None,
    tokens_punct: dict | None = None,
    subs_punct: dict | None = None,
) -> None:
    if not path.is_file():
        return
    for episode in _read_json(path).get("episodes", []):
        text = _text(episode)
        key = text_key(text)
        if not key or key in tokens_out:
            continue
        tokens = _tokens(episode)
        if tokens:
            tokens_out[key] = tokens
            if tokens_punct is not None:
                _remember_punct(tokens_punct, punct_key(text), tokens)
        if subs_out is None:
            continue
        rows = episode.get("sub_instruction_tokens")
        if rows:
            fitted = fit_sub_rows(rows)
            subs_out[key] = fitted
            if subs_punct is not None:
                _remember_punct(subs_punct, punct_key(text), fitted)


def load_official_instructions() -> OfficialInstructions:
    preprocessed = _dataset_dir("R2R_VLNCE_v1-3_preprocessed")
    fsa = _dataset_dir("R2R_VLNCE_FSASub")
    r2r = {}
    fsa_tokens = {}
    fsa_subs = {}
    fsa_tokens_punct = {}
    fsa_subs_punct = {}
    for split in _SPLITS:
        _index_split(preprocessed / split / f"{split}.json.gz", r2r, None)
        _index_split(
            fsa / split / f"{split}_sub.json.gz",
            fsa_tokens,
            fsa_subs,
            fsa_tokens_punct,
            fsa_subs_punct,
        )
    fsa_tokens_punct = {key: value for key, value in fsa_tokens_punct.items() if value is not _CONFLICT}
    fsa_subs_punct = {key: value for key, value in fsa_subs_punct.items() if value is not _CONFLICT}
    return OfficialInstructions(r2r, fsa_tokens, fsa_subs, fsa_tokens_punct, fsa_subs_punct)
