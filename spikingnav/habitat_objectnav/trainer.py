"""PPO loop that trains SpikingNav on Habitat HM3D ObjectNav."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from spikingnav.config import DEFAULT_CONFIG, SpikingNavConfig
from spikingnav.habitat_objectnav.ckpt_keeper import checkpoint_payload, restore_checkpoint
from spikingnav.habitat_objectnav.distributed import (
    RolloutPreemption,
    average_gradients,
    broadcast_parameters,
    distributed_context,
    init_distributed,
    normalize_distributed,
    reduce_mean,
    reduce_sum,
    rollout_steps_global,
    shutdown_distributed,
)
from spikingnav.habitat_objectnav.make_env import (
    DEFAULT_EPISODE_ROOT,
    DEFAULT_SCENES_DIR,
    HM3D_CATEGORIES,
    ensure_scene_dataset_config,
)
from spikingnav.models.actor_critic import SpikingNavModel

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _worker_env(split, scenes, gpu, seed, rank, max_episode_steps, episode_root, scenes_dir):
    from spikingnav.habitat_objectnav.make_env import (
        build_habitat_config,
        make_env_fn,
        scene_config_path,
    )

    config = build_habitat_config(
        split=split,
        scenes=scenes,
        gpu_device_id=gpu,
        max_episode_steps=max_episode_steps,
        episode_root=episode_root,
        scenes_dir=scenes_dir,
        seed=seed,
    )
    return make_env_fn(config, scene_config_path(scenes_dir), rank)


def _stack_obs(observations: Sequence[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    rgbs = []
    goals = []
    for obs in observations:
        rgb = np.asarray(obs["rgb"])
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        rgbs.append(rgb)
        goal = np.asarray(obs["objectgoal"]).reshape(-1)[0]
        goals.append(int(goal))
    return np.stack(rgbs, 0), np.asarray(goals, dtype=np.int64)


def _prepare_rgb(rgb_uint8: torch.Tensor, image_size: int) -> torch.Tensor:
    """uint8 NHWC -> ImageNet-normalized NCHW at ``image_size``."""
    x = rgb_uint8.float().permute(0, 3, 1, 2) / 255.0
    if x.shape[-1] != image_size or x.shape[-2] != image_size:
        x = torch.nn.functional.interpolate(
            x, size=(image_size, image_size), mode="bilinear", align_corners=False
        )
    mean = x.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (x - mean) / std


class HabitatSpikingTrainer:
    def __init__(
        self,
        num_envs: int = 4,
        split: str = "train",
        scenes: Optional[Sequence[str]] = None,
        gpu: int = 0,
        seed: int = 12345,
        rollout_steps: int = 128,
        total_steps: int = 300_000_000,
        max_episode_steps: int = 500,
        pretrained_ann: bool = True,
        output_dir: str = "storage/habitat-objectnav-hm3d-spiking",
        log_interval: int = 10,
        save_interval_updates: int = 50,
        episode_root: str = DEFAULT_EPISODE_ROOT,
        scenes_dir: str = DEFAULT_SCENES_DIR,
        preemption_threshold: float = 0.6,
        resume_path: str = "",
        cfg: Optional[SpikingNavConfig] = None,
    ) -> None:
        self.cfg = cfg or DEFAULT_CONFIG
        self.rollout_steps = rollout_steps
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.save_interval_updates = save_interval_updates
        self.output_dir = output_dir
        self.rank, self.world_size, local_rank = distributed_context()
        self.is_main = self.rank == 0
        if self.world_size > 1:
            gpu = local_rank
            seed = int(seed) + self.rank * 10000
        init_distributed(local_rank, self.world_size)
        if self.world_size > 1:
            self.device = torch.device("cuda", local_rank)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(output_dir, exist_ok=True)
        ensure_scene_dataset_config(scenes_dir)

        from habitat.core.vector_env import VectorEnv

        env_fn_args = tuple(
            (
                split,
                list(scenes) if scenes else None,
                gpu,
                seed,
                self.rank * num_envs + rank,
                max_episode_steps,
                episode_root,
                scenes_dir,
            )
            for rank in range(num_envs)
        )
        self.envs = VectorEnv(
            make_env_fn=_worker_env,
            env_fn_args=env_fn_args,
            auto_reset_done=True,
            multiprocessing_start_method="forkserver",
        )
        self.num_envs = self.envs.num_envs
        action_space = self.envs.action_spaces[0]
        action_dim = int(action_space.n)
        self.model = SpikingNavModel(
            action_dim=action_dim,
            num_categories=len(HM3D_CATEGORIES),
            cfg=self.cfg,
            pretrained_ann=pretrained_ann,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)
        self.resume_steps = 0
        self.resume_optimizer = False
        if resume_path:
            self.resume_steps, self.resume_optimizer = restore_checkpoint(
                resume_path, self.model, self.optimizer, self.device
            )
        broadcast_parameters(self.model)
        self.memory = self.model.initial_memory(self.num_envs, self.device)
        self.masks = torch.ones(self.num_envs, 1, device=self.device)
        self.preemption = RolloutPreemption(preemption_threshold)

    def _global_rollout_steps(self) -> int:
        return rollout_steps_global(self.rollout_steps, self.num_envs, self.world_size)

    def _writer(self):
        if not self.is_main:
            return None
        from torch.utils.tensorboard import SummaryWriter

        tb_dir = os.path.join(self.output_dir, "tb")
        print(f"tensorboard {tb_dir}", flush=True)
        if self.world_size > 1:
            print(
                f"distributed ranks {self.world_size}, {self.num_envs} envs each",
                flush=True,
            )
        return SummaryWriter(tb_dir)

    def _finish(self, writer) -> None:
        if writer is not None:
            writer.close()
        shutdown_distributed()

    def close(self) -> None:
        self.envs.close()

    def _policy_step(self, rgb_uint8: torch.Tensor, goals: torch.Tensor):
        rgb = _prepare_rgb(rgb_uint8, self.cfg.image_size)
        logits, values, self.memory = self.model(
            rgb, goals, self.memory, self.masks.view(1, -1, 1)
        )
        logits = logits[-1]
        values = values[-1].squeeze(-1)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        return actions, dist.log_prob(actions), values, dist.entropy()

    def collect(self, observations: List[Dict], update_idx: int = 0):
        cfg = self.cfg
        n = self.num_envs
        t_steps = self.rollout_steps
        sample_rgb, sample_goal = _stack_obs(observations)
        h, w = sample_rgb.shape[1:3]
        rgb_buf = torch.zeros(t_steps, n, h, w, 3, dtype=torch.uint8, device=self.device)
        goal_buf = torch.zeros(t_steps, n, dtype=torch.long, device=self.device)
        action_buf = torch.zeros(t_steps, n, dtype=torch.long, device=self.device)
        log_prob_buf = torch.zeros(t_steps, n, device=self.device)
        value_buf = torch.zeros(t_steps, n, device=self.device)
        reward_buf = torch.zeros(t_steps, n, device=self.device)
        mask_buf = torch.zeros(t_steps, n, 1, device=self.device)
        done_buf = torch.zeros(t_steps, n, device=self.device)
        success_count = 0.0
        spl_count = 0.0
        episode_count = 0

        t_done = 0
        for t in range(t_steps):
            if self.preemption.should_stop(update_idx, t_done, t_steps):
                break
            rgb_np, goal_np = _stack_obs(observations)
            rgb_buf[t] = torch.from_numpy(rgb_np).to(self.device)
            goal_buf[t] = torch.from_numpy(goal_np).to(self.device)
            mask_buf[t] = self.masks
            with torch.no_grad():
                actions, log_probs, values, _ = self._policy_step(rgb_buf[t], goal_buf[t])
            action_buf[t] = actions
            log_prob_buf[t] = log_probs
            value_buf[t] = values
            step_result = self.envs.step(actions.detach().cpu().numpy())
            observations = [item[0] for item in step_result]
            rewards = np.asarray([item[1] for item in step_result], dtype=np.float32)
            dones = np.asarray([item[2] for item in step_result], dtype=np.bool_)
            infos = [item[3] for item in step_result]
            reward_buf[t] = torch.from_numpy(rewards).to(self.device)
            done_buf[t] = torch.from_numpy(dones.astype(np.float32)).to(self.device)
            self.masks = torch.from_numpy((~dones).astype(np.float32)).to(self.device).unsqueeze(-1)
            self.memory = self.memory * self.masks
            for info, done in zip(infos, dones):
                if done:
                    episode_count += 1
                    success_count += float(info.get("success", 0.0))
                    spl_count += float(info.get("spl", 0.0))
            t_done = t + 1
        else:
            self.preemption.mark_finished(update_idx)
        self.preemption.barrier()
        with torch.no_grad():
            rgb_np, goal_np = _stack_obs(observations)
            _, _, next_value, _ = self._policy_step(
                torch.from_numpy(rgb_np).to(self.device),
                torch.from_numpy(goal_np).to(self.device),
            )
        kept = slice(0, t_done)
        return {
            "rgb": rgb_buf[kept],
            "goal": goal_buf[kept],
            "action": action_buf[kept],
            "log_prob": log_prob_buf[kept],
            "value": value_buf[kept],
            "reward": reward_buf[kept],
            "mask": mask_buf[kept],
            "done": done_buf[kept],
            "next_value": next_value.detach(),
            "success_count": success_count,
            "spl_count": spl_count,
            "episode_count": episode_count,
            "observations": observations,
            "steps_done": t_done,
        }

    def _gae(self, reward, value, done, next_value):
        cfg = self.cfg
        advantages = torch.zeros_like(reward)
        last_gae = torch.zeros(reward.shape[1], device=reward.device)
        next_val = next_value
        for t in reversed(range(reward.shape[0])):
            next_non_terminal = 1.0 - done[t]
            delta = reward[t] + cfg.gamma * next_val * next_non_terminal - value[t]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
            next_val = value[t]
        returns = advantages + value
        return advantages, returns

    def update(self, batch) -> Dict[str, float]:
        cfg = self.cfg
        advantages, returns = self._gae(
            batch["reward"], batch["value"], batch["done"], batch["next_value"]
        )
        advantages = normalize_distributed(advantages)
        t_steps, n = batch["action"].shape
        total_loss = 0.0
        for _ in range(cfg.update_repeats):
            self.optimizer.zero_grad(set_to_none=True)
            rgb = _prepare_rgb(batch["rgb"].reshape(t_steps * n, *batch["rgb"].shape[2:]), cfg.image_size)
            # Recompute the rollout with stored masks so the membrane matches collection.
            memory = torch.zeros(n, cfg.hidden_size, device=self.device)
            logits_all = []
            values_all = []
            for t in range(t_steps):
                start = t * n
                logits, values, memory = self.model(
                    rgb[start : start + n],
                    batch["goal"][t],
                    memory,
                    batch["mask"][t].view(1, -1, 1),
                )
                logits_all.append(logits[-1])
                values_all.append(values[-1].squeeze(-1))
            logits = torch.stack(logits_all, 0)
            values = torch.stack(values_all, 0)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(batch["action"])
            entropy = dist.entropy().mean()
            ratio = torch.exp(log_probs - batch["log_prob"])
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            # AllenAct PPO clips the value target with the same ε as the policy.
            value_pred_clipped = batch["value"] + (values - batch["value"]).clamp(
                -cfg.clip_param, cfg.clip_param
            )
            value_losses = (values - returns).pow(2)
            value_losses_clipped = (value_pred_clipped - returns).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
            loss = policy_loss + cfg.value_loss_coef * value_loss - cfg.entropy_coef * entropy
            loss.backward()
            sample_count = float(t_steps * n)
            average_gradients(self.model, sample_count)
            clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
            self.optimizer.step()
            total_loss += float(loss.detach())
            last = {
                "policy_loss": float(policy_loss.detach()),
                "value_loss": float(value_loss.detach()),
                "entropy": float(entropy.detach()),
            }
        last["loss"] = total_loss / cfg.update_repeats
        return last

    def train(self) -> None:
        writer = self._writer()
        if self.is_main:
            print(
                "reward r_t = 10 * success + clip(geodesic decrease, ±motion) - 0.01; "
                "PPO clip 0.1, value 0.5, entropy 0.01, lr 3e-4 linear decay",
                flush=True,
            )
            if self.world_size > 1:
                print(
                    f"DD-PPO world {self.world_size}, preemption {self.preemption.threshold} "
                    f"(a worker may stop after {self.preemption.min_fraction:.0%} of the rollout)",
                    flush=True,
                )
        observations = self.envs.reset()
        steps = self.resume_steps
        if self.is_main and self.resume_steps:
            optimizer_note = "optimizer restored" if self.resume_optimizer else "optimizer starts fresh"
            print(f"resume from step {steps}, {optimizer_note}", flush=True)
        update_idx = 0
        success_acc = 0.0
        spl_acc = 0.0
        episode_acc = 0
        try:
            while steps < self.total_steps:
                batch = self.collect(observations, update_idx)
                observations = batch.pop("observations")
                steps_done = int(batch.pop("steps_done"))
                success_acc += batch.pop("success_count")
                spl_acc += batch.pop("spl_count")
                episode_acc += batch.pop("episode_count")
                reward_mean = float(batch["reward"].mean())
                progress = min(1.0, steps / float(self.total_steps))
                lr = self.cfg.learning_rate * (1.0 - progress)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr
                stats = self.update(batch)
                steps += int(round(reduce_sum(float(steps_done * self.num_envs), self.device)))
                update_idx += 1
                if update_idx % self.log_interval == 0:
                    success_acc = reduce_sum(success_acc, self.device)
                    spl_acc = reduce_sum(spl_acc, self.device)
                    episode_acc = int(round(reduce_sum(float(episode_acc), self.device)))
                    reward_mean = reduce_mean(reward_mean, self.device)
                    rollout_mean = reduce_mean(float(steps_done), self.device)
                    for key in ("loss", "policy_loss", "value_loss", "entropy"):
                        stats[key] = reduce_mean(stats[key], self.device)
                    if self.is_main:
                        sr = success_acc / episode_acc if episode_acc else 0.0
                        spl = spl_acc / episode_acc if episode_acc else 0.0
                        print(
                            f"update {update_idx} steps {steps} loss {stats['loss']:.4f} "
                            f"episodes {episode_acc} success {sr:.4f} spl {spl:.4f}",
                            flush=True,
                        )
                        writer.add_scalar("train/loss", stats["loss"], steps)
                        writer.add_scalar("train/policy_loss", stats["policy_loss"], steps)
                        writer.add_scalar("train/value_loss", stats["value_loss"], steps)
                        writer.add_scalar("train/entropy", stats["entropy"], steps)
                        writer.add_scalar("train/reward", reward_mean, steps)
                        writer.add_scalar("train/lr", lr, steps)
                        writer.add_scalar("train/rollout_steps", rollout_mean, steps)
                        writer.add_scalar("train/episodes", episode_acc, steps)
                        if episode_acc:
                            writer.add_scalar("train/success", sr, steps)
                            writer.add_scalar("train/spl", spl, steps)
                        writer.flush()
                    success_acc = 0.0
                    spl_acc = 0.0
                    episode_acc = 0
                if update_idx % self.save_interval_updates == 0 and self.is_main:
                    path = os.path.join(self.output_dir, f"ckpt_steps_{steps}.pt")
                    torch.save(
                        checkpoint_payload(self.model, self.optimizer, steps, algo="ppo"),
                        path,
                    )
                    print(f"saved {path}", flush=True)
        finally:
            self._finish(writer)
