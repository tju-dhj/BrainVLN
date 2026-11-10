"""Behavior cloning on stored human demonstrations. No simulator."""

from __future__ import annotations

import glob
import os
from typing import Dict, List

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

from spikingnav.config import DEFAULT_CONFIG, SpikingNavConfig
from spikingnav.habitat_objectnav.il_trainer import imitation_loss
from spikingnav.habitat_objectnav.make_env import HM3D_CATEGORIES
from spikingnav.habitat_objectnav.offline_store import load_episode
from spikingnav.habitat_objectnav.trainer import _prepare_rgb
from spikingnav.models.actor_critic import SpikingNavModel


class DemoDataset(Dataset):
    def __init__(self, root: str, window: int) -> None:
        self.paths = sorted(glob.glob(os.path.join(root, "**", "*.npz"), recursive=True))
        if not self.paths:
            raise FileNotFoundError(f"no episodes under {root}")
        self.window = int(window)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        episode = load_episode(self.paths[index])
        length = int(episode["actions"].shape[0])
        window = min(self.window, length)
        start = 0 if length == window else int(np.random.randint(0, length - window + 1))
        end = start + window
        mask = np.ones((window, 1), dtype=np.float32)
        if start == 0:
            mask[0, 0] = 0.0
        return {
            "rgb": torch.from_numpy(episode["rgb"][start:end]),
            "goal": torch.full((window,), int(episode["goal"]), dtype=torch.long),
            "action": torch.from_numpy(episode["actions"][start:end]),
            "mask": torch.from_numpy(mask),
        }


def _pad_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    length = max(item["action"].shape[0] for item in batch)
    rgb = torch.zeros(length, len(batch), *batch[0]["rgb"].shape[1:], dtype=torch.uint8)
    goal = torch.zeros(length, len(batch), dtype=torch.long)
    action = torch.zeros(length, len(batch), dtype=torch.long)
    mask = torch.zeros(length, len(batch), 1)
    valid = torch.zeros(length, len(batch))
    for index, item in enumerate(batch):
        steps = item["action"].shape[0]
        rgb[:steps, index] = item["rgb"]
        goal[:steps, index] = item["goal"]
        action[:steps, index] = item["action"]
        mask[:steps, index] = item["mask"]
        valid[:steps, index] = 1.0
    return {"rgb": rgb, "goal": goal, "action": action, "mask": mask, "valid": valid}


def train_offline(
    root: str,
    output_dir: str,
    steps: int,
    batch_size: int = 8,
    window: int = 32,
    lr: float = 3e-4,
    log_interval: int = 10,
    pretrained_ann: bool = True,
    cfg: SpikingNavConfig = None,
) -> None:
    cfg = cfg or DEFAULT_CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DemoDataset(root, window)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=_pad_collate,
        drop_last=True,
        persistent_workers=True,
    )
    model = SpikingNavModel(
        action_dim=6,
        num_categories=len(HM3D_CATEGORIES),
        cfg=cfg,
        pretrained_ann=pretrained_ann,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    os.makedirs(output_dir, exist_ok=True)
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(os.path.join(output_dir, "tb"))
    print(f"offline episodes {len(dataset)} tensorboard {output_dir}/tb", flush=True)
    step = 0
    while step < steps:
        for batch in loader:
            rgb = _prepare_rgb(
                batch["rgb"].reshape(-1, *batch["rgb"].shape[2:]).to(device),
                cfg.image_size,
            )
            t_steps, n = batch["action"].shape
            rgb = rgb.view(t_steps, n, *rgb.shape[1:])
            goal = batch["goal"].to(device)
            memory = torch.zeros(n, cfg.hidden_size, device=device)
            logits_all = []
            for t in range(t_steps):
                logits, _, memory = model(
                    rgb[t], goal[t], memory, batch["mask"][t].to(device).view(1, -1, 1)
                )
                logits_all.append(logits[-1])
            loss = imitation_loss(
                torch.stack(logits_all, 0),
                batch["action"].to(device),
                batch["valid"].to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            step += n * t_steps
            if step % (log_interval * n * t_steps) < n * t_steps:
                print(f"steps {step} loss {float(loss):.4f}", flush=True)
                writer.add_scalar("train/bc_loss", float(loss), step)
                writer.flush()
            if step >= steps:
                break
    path = os.path.join(output_dir, f"ckpt_steps_{step}.pt")
    torch.save({"model": model.state_dict(), "steps": step}, path)
    writer.close()
    print(f"saved {path}", flush=True)
