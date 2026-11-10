"""Scene-grouped dataset over read-only collected R2R-CE episodes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from torch.utils.data import Dataset

from spikingnav.brain_vln.data.contract import (
    BURN_IN_STEPS,
    HOLDOUT_SCENE_COUNT,
    TRUNCATE_EVERY,
    WINDOW_SIZE,
    DataContractError,
    assign_scenes,
)
from spikingnav.brain_vln.data.episode import EpisodeRecord, load_episode_record, read_header
from spikingnav.brain_vln.data.sampler import WindowSpec, plan_windows
from spikingnav.brain_vln.data.transforms import SyncRGBDTransform


@dataclass(frozen=True)
class _SampleRef:
    episode_dir: Path
    start: int
    valid_length: int
    burn_in_steps: int


def scene_names(root: Path) -> List[str]:
    root = Path(root)
    if not root.is_dir():
        raise DataContractError(f"data root does not exist: {root}")
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def ready_episode_dirs(scene_dir: Path) -> List[Path]:
    """Stable episode order by directory name. This order is not frame time."""
    return sorted(
        path for path in scene_dir.iterdir() if path.is_dir() and (path / "READY").is_file()
    )


def select_spotcheck_episodes(root: Path, count: int = 6) -> List[Path]:
    """Pick the first READY episode of scenes spread across the sorted scene list."""
    if count < 1:
        raise DataContractError("spot check count must be positive")
    scenes = [Path(root) / name for name in scene_names(root)]
    if len(scenes) < count:
        raise DataContractError(f"need at least {count} scenes, found {len(scenes)}")
    if count == 1:
        positions = [0]
    else:
        positions = [round(index * (len(scenes) - 1) / (count - 1)) for index in range(count)]
    chosen: List[Path] = []
    for position in positions:
        episodes = ready_episode_dirs(scenes[position])
        if not episodes:
            raise DataContractError(f"no READY episode in {scenes[position].name}")
        chosen.append(episodes[0])
    return chosen


class BrainVLNDataset(Dataset):
    """Windows or full episodes from an internal scene split of the train collection.

    ``split='holdout'`` is a development split of collected train scenes. It is
    not official Val-Seen or Val-Unseen.
    """

    def __init__(
        self,
        root: Path,
        split: str = "train",
        mode: str = "window",
        window_size: int = WINDOW_SIZE,
        burn_in: int = BURN_IN_STEPS,
        truncate_every: int = TRUNCATE_EVERY,
        holdout_count: int = HOLDOUT_SCENE_COUNT,
        transform: Optional[SyncRGBDTransform] = None,
        scene_filter: Optional[Sequence[str]] = None,
    ) -> None:
        if split not in ("train", "holdout", "all"):
            raise DataContractError("split must be train, holdout, or all")
        if mode not in ("window", "episode"):
            raise DataContractError("mode must be window or episode")
        self.root = Path(root)
        self.split = split
        self.mode = mode
        self.window_size = window_size
        self.burn_in = burn_in
        self.truncate_every = truncate_every
        self.transform = transform
        names = scene_names(self.root)
        if scene_filter is not None:
            unknown = sorted(set(scene_filter) - set(names))
            if unknown:
                raise DataContractError(f"unknown scenes {unknown}")
            names = [name for name in names if name in set(scene_filter)]
        self.assignment = assign_scenes(names, holdout_count)
        selected = names if split == "all" else list(self.assignment[split])
        self.samples: List[_SampleRef] = []
        for scene in selected:
            for episode_dir in ready_episode_dirs(self.root / scene):
                header = read_header(episode_dir)
                specs = (
                    [WindowSpec(0, header.steps, 0)]
                    if mode == "episode"
                    else plan_windows(header.steps, window_size, burn_in)
                )
                for spec in specs:
                    self.samples.append(
                        _SampleRef(episode_dir, spec.start, spec.valid_length, spec.burn_in_steps)
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> EpisodeRecord:
        sample = self.samples[index]
        return load_episode_record(
            sample.episode_dir,
            start=sample.start,
            valid_length=sample.valid_length,
            burn_in_steps=sample.burn_in_steps,
            pad_to=self.window_size if self.mode == "window" else None,
            truncate_every=self.truncate_every,
            transform=self.transform,
        )
