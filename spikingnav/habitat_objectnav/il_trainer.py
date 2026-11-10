"""DAgger imitation learning for Habitat HM3D ObjectNav.

The student is supervised by the shortest-path expert in ``expert.py``.
``beta`` is the probability of executing the expert action. It starts at 1
(behavior cloning on expert states) and decays toward ``beta_end`` so later
updates also label states the student visits itself.
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from spikingnav.habitat_objectnav.ckpt_keeper import checkpoint_payload
from spikingnav.habitat_objectnav.distributed import average_gradients, reduce_mean, reduce_sum
from spikingnav.habitat_objectnav.expert import dagger_beta
from spikingnav.habitat_objectnav.trainer import (
    HabitatSpikingTrainer,
    _prepare_rgb,
    _stack_obs,
)


def imitation_loss(
    logits: torch.Tensor, expert: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Cross-entropy of the student against expert actions.

    ``logits`` is ``[T, N, A]``, ``expert`` and ``valid`` are ``[T, N]``.
    Invalid expert steps are left out of the mean.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    chosen = log_probs.gather(-1, expert.long().unsqueeze(-1)).squeeze(-1)
    weight = valid.float()
    return -(chosen * weight).sum() / weight.sum().clamp(min=1.0)


class HabitatILTrainer(HabitatSpikingTrainer):
    def __init__(
        self,
        *args,
        beta_start: float = 1.0,
        beta_end: float = 0.0,
        beta_decay_steps: int = 0,
        expert: str = "geodesic",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if expert not in ("geodesic", "human"):
            raise ValueError(f"unknown expert {expert}")
        self.expert = expert
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.beta_decay_steps = int(beta_decay_steps)
        self._env_steps = self.resume_steps
        self._expert_call = "demo_action" if expert == "human" else "expert_action"

    def _beta(self) -> float:
        return dagger_beta(
            self._env_steps,
            self.total_steps,
            self.beta_start,
            self.beta_end,
            self.beta_decay_steps,
        )

    def collect(self, observations):
        n = self.num_envs
        t_steps = self.rollout_steps
        sample_rgb, _ = _stack_obs(observations)
        h, w = sample_rgb.shape[1:3]
        rgb_buf = torch.zeros(t_steps, n, h, w, 3, dtype=torch.uint8, device=self.device)
        goal_buf = torch.zeros(t_steps, n, dtype=torch.long, device=self.device)
        expert_buf = torch.zeros(t_steps, n, dtype=torch.long, device=self.device)
        valid_buf = torch.zeros(t_steps, n, device=self.device)
        mask_buf = torch.zeros(t_steps, n, 1, device=self.device)
        taken_buf = torch.zeros(t_steps, n, dtype=torch.long, device=self.device)
        memory0 = self.memory.detach().clone()
        success_count = 0.0
        spl_count = 0.0
        episode_count = 0
        beta = self._beta()

        for t in range(t_steps):
            rgb_np, goal_np = _stack_obs(observations)
            rgb_buf[t] = torch.from_numpy(rgb_np).to(self.device)
            goal_buf[t] = torch.from_numpy(goal_np).to(self.device)
            mask_buf[t] = self.masks
            queried = self.envs.call([self._expert_call] * n)
            expert = np.asarray([item[0] for item in queried], dtype=np.int64)
            valid = np.asarray([item[1] for item in queried], dtype=np.float32)
            expert_buf[t] = torch.from_numpy(expert).to(self.device)
            valid_buf[t] = torch.from_numpy(valid).to(self.device)
            with torch.no_grad():
                student, _, _, _ = self._policy_step(rgb_buf[t], goal_buf[t])
            use_expert = (np.random.rand(n) < beta) & (valid > 0.5)
            actions = student.detach().clone()
            if np.any(use_expert):
                expert_t = torch.from_numpy(expert).to(self.device)
                pick = torch.from_numpy(use_expert).to(self.device)
                actions = torch.where(pick, expert_t, actions)
            taken_buf[t] = actions
            step_result = self.envs.step(actions.detach().cpu().numpy())
            observations = [item[0] for item in step_result]
            dones = np.asarray([item[2] for item in step_result], dtype=np.bool_)
            infos = [item[3] for item in step_result]
            self.masks = torch.from_numpy((~dones).astype(np.float32)).to(self.device).unsqueeze(-1)
            self.memory = self.memory * self.masks
            for info, done in zip(infos, dones):
                if done:
                    episode_count += 1
                    success_count += float(info.get("success", 0.0))
                    spl_count += float(info.get("spl", 0.0))
        return {
            "rgb": rgb_buf,
            "goal": goal_buf,
            "expert": expert_buf,
            "valid": valid_buf,
            "mask": mask_buf,
            "taken": taken_buf,
            "memory0": memory0,
            "beta": beta,
            "success_count": success_count,
            "spl_count": spl_count,
            "episode_count": episode_count,
            "observations": observations,
        }

    def update(self, batch) -> Dict[str, float]:
        cfg = self.cfg
        t_steps, n = batch["expert"].shape
        total_loss = 0.0
        last = {"loss": 0.0, "agreement": 0.0}
        for _ in range(cfg.update_repeats):
            self.optimizer.zero_grad(set_to_none=True)
            rgb = _prepare_rgb(
                batch["rgb"].reshape(t_steps * n, *batch["rgb"].shape[2:]),
                cfg.image_size,
            )
            memory = batch["memory0"].detach().clone()
            logits_all = []
            for t in range(t_steps):
                start = t * n
                logits, _, memory = self.model(
                    rgb[start : start + n],
                    batch["goal"][t],
                    memory,
                    batch["mask"][t].view(1, -1, 1),
                )
                logits_all.append(logits[-1])
            logits = torch.stack(logits_all, 0)
            loss = imitation_loss(logits, batch["expert"], batch["valid"])
            loss.backward()
            average_gradients(self.model)
            clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
            self.optimizer.step()
            with torch.no_grad():
                match = (logits.argmax(-1) == batch["expert"]).float()
                agree = (match * batch["valid"]).sum() / batch["valid"].sum().clamp(min=1.0)
            total_loss += float(loss.detach())
            last = {"loss": float(loss.detach()), "agreement": float(agree)}
        last["loss"] = total_loss / cfg.update_repeats
        return last

    def train(self) -> None:
        writer = self._writer()
        if self.is_main:
            print(
                f"il expert {self.expert} beta {self.beta_start:.2f} -> {self.beta_end:.2f}",
                flush=True,
            )
        observations = self.envs.reset()
        if self.is_main and self.resume_steps:
            optimizer_note = "optimizer restored" if self.resume_optimizer else "optimizer starts fresh"
            print(
                f"resume from step {self._env_steps}, {optimizer_note}",
                flush=True,
            )
        update_idx = 0
        success_acc = 0.0
        spl_acc = 0.0
        episode_acc = 0
        try:
            while self._env_steps < self.total_steps:
                batch = self.collect(observations)
                observations = batch.pop("observations")
                success_acc += batch.pop("success_count")
                spl_acc += batch.pop("spl_count")
                episode_acc += batch.pop("episode_count")
                beta = float(batch.pop("beta"))
                stats = self.update(batch)
                self._env_steps += self._global_rollout_steps()
                update_idx += 1
                if update_idx % self.log_interval == 0:
                    success_acc = reduce_sum(success_acc, self.device)
                    spl_acc = reduce_sum(spl_acc, self.device)
                    episode_acc = int(round(reduce_sum(float(episode_acc), self.device)))
                    beta = reduce_mean(beta, self.device)
                    for key in ("loss", "agreement"):
                        stats[key] = reduce_mean(stats[key], self.device)
                    if self.is_main:
                        sr = success_acc / episode_acc if episode_acc else 0.0
                        spl = spl_acc / episode_acc if episode_acc else 0.0
                        print(
                            f"update {update_idx} steps {self._env_steps} "
                            f"bc {stats['loss']:.4f} agree {stats['agreement']:.3f} "
                            f"beta {beta:.3f} episodes {episode_acc} "
                            f"success {sr:.4f} spl {spl:.4f}",
                            flush=True,
                        )
                        writer.add_scalar("train/bc_loss", stats["loss"], self._env_steps)
                        writer.add_scalar("train/agreement", stats["agreement"], self._env_steps)
                        writer.add_scalar("train/beta", beta, self._env_steps)
                        writer.add_scalar("train/episodes", episode_acc, self._env_steps)
                        if episode_acc:
                            writer.add_scalar("train/success", sr, self._env_steps)
                            writer.add_scalar("train/spl", spl, self._env_steps)
                        writer.flush()
                    success_acc = 0.0
                    spl_acc = 0.0
                    episode_acc = 0
                if update_idx % self.save_interval_updates == 0 and self.is_main:
                    path = os.path.join(self.output_dir, f"ckpt_steps_{self._env_steps}.pt")
                    torch.save(
                        checkpoint_payload(
                            self.model, self.optimizer, self._env_steps, algo="il"
                        ),
                        path,
                    )
                    print(f"saved {path}", flush=True)
        finally:
            self._finish(writer)
