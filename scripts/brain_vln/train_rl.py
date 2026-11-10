#!/usr/bin/env python3
"""PPO fine-tune of BrainVLN_v2 from the best finished imitation checkpoint.

Edit ``configs/brain_vln_rl.yaml``. The defaults match MLANet's released
``mlanet/config/mla/mla_ppo.yaml``. The policy is loaded from imitation. The
critic and Adam state start fresh, because an imitation checkpoint has no
value head. An empty ``init`` selects the finished val-unseen summary with
the highest SPL. The script does not start until invoked, and it refuses to
overwrite an existing output directory.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

from spikingnav.brain_vln.closed_loop import score_trajectory
from spikingnav.brain_vln.config import BrainVLNConfig, PopulationSizes
from spikingnav.brain_vln.data.contract import assign_scenes
from spikingnav.brain_vln.data.dataset import ready_episode_dirs, scene_names
from spikingnav.brain_vln.il_checkpoint import checkpoint_due, select_best_il_checkpoint, shard_episodes
from spikingnav.brain_vln.model import BrainVLNModel
from spikingnav.brain_vln.ppo import RecurrentActor, RolloutStep, ValueHead, ppo_update
from spikingnav.brain_vln.reward import MLANetReward
from spikingnav.brain_vln.rl_config import settings_from_config
from spikingnav.brain_vln.rl_env import LowLevelEnv, load_episode_task
from spikingnav.vlnce.constants import STOP
from spikingnav.vlnce.episodes import mesh_path


def _config_from_checkpoint(payload: dict) -> BrainVLNConfig:
    raw = dict(payload["config"])
    raw.pop("neuron_count", None)
    populations = PopulationSizes(**raw.pop("populations"))
    config = BrainVLNConfig(**raw)
    config.populations = populations
    return config


def _train_episodes(root: Path) -> list:
    names = scene_names(root)
    chosen = set(assign_scenes(names)["train"])
    episodes = []
    for name in sorted(chosen):
        episodes.extend(ready_episode_dirs(root / name))
    if not episodes:
        raise RuntimeError(f"no READY train episodes under {root}")
    return episodes


def _finite_score(score: dict) -> dict:
    cleaned = {}
    for key, value in score.items():
        if isinstance(value, float) and not math.isfinite(value):
            cleaned[key] = None
        else:
            cleaned[key] = value
    return cleaned


def _save(path: Path, model, critic, optimizer, update: int, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "critic": critic.state_dict(),
            "optimizer": optimizer.state_dict(),
            "update": update,
            "config": model.config.to_dict(),
            "record": record,
        },
        path,
    )


class EnvironmentWorker:
    """One simulator and one reservoir stream. Episodes reset inside a rollout."""

    def __init__(
        self,
        episodes: list,
        actor: RecurrentActor,
        reward: MLANetReward,
        scenes: Path,
        sim_gpu: int,
        max_steps: int,
    ) -> None:
        if not episodes:
            raise ValueError("an environment needs at least one episode")
        self.episodes = list(episodes)
        self.cursor = 0
        self.actor = actor
        self.reward = reward
        self.scenes = scenes
        self.sim_gpu = int(sim_gpu)
        self.max_steps = int(max_steps)
        self.sim = None
        self.scene_id = None
        self.env = None
        self.task = None
        self.needs_reset = True
        self.episode_return = 0.0

    def close(self) -> None:
        if self.sim is not None:
            self.sim.close()
            self.sim = None

    def _start_episode(self) -> None:
        from spikingnav.vlnce.sim import make_simulator

        episode_dir = self.episodes[self.cursor % len(self.episodes)]
        self.cursor += 1
        self.task = load_episode_task(episode_dir)
        if self.task["scene_id"] != self.scene_id:
            if self.sim is not None:
                self.sim.close()
            mesh = mesh_path(str(self.scenes), self.task["scene_id"])
            self.sim = make_simulator(str(mesh), self.sim_gpu)
            self.scene_id = self.task["scene_id"]
        self.env = LowLevelEnv(self.sim, self.task["goal"], self.reward, self.max_steps)
        self.env.reset(self.task["start_position"], self.task["start_rotation"])
        self.actor.reset()
        self.actor.observe(
            self.env.observation(),
            self.task["tokens"],
            self.task["token_mask"],
            self.task["instruction_text"],
        )
        self.needs_reset = False
        self.episode_return = 0.0

    def rollout(self, rollout_steps: int) -> tuple:
        if rollout_steps < 1:
            raise ValueError("rollout_steps must be positive")
        steps = []
        finished = []
        for _ in range(rollout_steps):
            if self.needs_reset:
                self._start_episode()
            action, log_prob = self.actor.sample()
            cached = self.actor.cache
            _observation, reward, done, _info = self.env.step(action)
            self.episode_return += float(reward)
            steps.append(
                RolloutStep(
                    rgb=cached["rgb"],
                    depth=cached["depth"],
                    depth_valid=cached["depth_valid"],
                    tokens=cached["tokens"],
                    token_mask=cached["token_mask"],
                    instruction_text=cached["instruction_text"],
                    prev_action=cached["prev_action"],
                    action=int(action),
                    old_log_prob=float(log_prob),
                    value=float(cached["value"]),
                    reward=float(reward),
                    done=bool(done),
                    reset=bool(cached["reset"]),
                    state=cached["state"],
                )
            )
            self.actor.commit(action)
            if done:
                finished.append(self._finish(int(action)))
                self.needs_reset = True
                self.actor.reset()
            else:
                self.actor.observe(
                    self.env.observation(),
                    self.task["tokens"],
                    self.task["token_mask"],
                    self.task["instruction_text"],
                )
        bootstrap = 0.0 if steps[-1].done else float(self.actor.cache["value"])
        return steps, bootstrap, finished

    def _finish(self, action: int) -> dict:
        start = self.env.start_distance
        score = score_trajectory(
            self.env.positions,
            self.task["reference_path"],
            action == STOP,
            float(self.env.distance),
            float(start) if math.isfinite(start) else -1.0,
            self.reward.success_distance,
        )
        return {
            "episode_id": self.task["episode_id"],
            "scene": self.task["scene"],
            "return": self.episode_return,
            "episode": _finite_score(score),
        }


def train(cfg) -> None:
    args = settings_from_config(cfg)
    args.output = Path(args.output)
    args.data = Path(args.data)
    args.scenes = Path(args.scenes)
    args.eval_root = Path(args.eval_root)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.init is None:
        try:
            init_path, chosen = select_best_il_checkpoint(args.eval_root)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc
    else:
        init_path = Path(args.init)
        chosen = None
        if not init_path.is_file():
            raise SystemExit(f"imitation checkpoint not found: {init_path}")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    payload = torch.load(init_path, map_location="cpu")
    config = _config_from_checkpoint(payload)
    model = BrainVLNModel(config).to(device)
    model.load_state_dict(payload["model"])
    model.frontend.set_frozen(True, True, True)
    critic = ValueHead(config.populations.action + config.populations.goal).to(device)
    base_ppo = args.ppo
    trainable = [param for param in list(model.parameters()) + list(critic.parameters()) if param.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=base_ppo.learning_rate, eps=base_ppo.adam_eps)
    reward = args.reward
    episodes = _train_episodes(args.data)
    shards = shard_episodes(episodes, args.environments)
    workers = [
        EnvironmentWorker(
            shard,
            RecurrentActor(model, critic),
            reward,
            args.scenes,
            args.sim_gpu,
            args.max_episode_steps,
        )
        for shard in shards
    ]
    args.output.mkdir(parents=True)
    schedule = {
        "config_file": "configs/brain_vln_rl.yaml",
        "source": "RavenKiller/MLANet mlanet/config/mla/mla_ppo.yaml",
        "init": str(init_path),
        "selection": "highest finished val-unseen SPL" if chosen is not None else "explicit init",
        "selected_spl": None if chosen is None else chosen.get("spl"),
        "selected_sr": None if chosen is None else chosen.get("sr"),
        "selected_update": None if chosen is None else chosen.get("update"),
        "critic": "fresh zero value head; imitation checkpoints have no critic",
        "optimizer": "new Adam; imitation optimizer state is not loaded",
        "environments": args.environments,
        "updates": args.updates,
        "rollout_steps": args.rollout_steps,
        "ppo_epochs": base_ppo.epochs,
        "mini_batches": args.mini_batches,
        "clip": base_ppo.clip,
        "learning_rate": base_ppo.learning_rate,
        "linear_lr_decay": args.linear_lr_decay,
        "linear_clip_decay": args.linear_clip_decay,
        "value_coef": base_ppo.value_coef,
        "entropy_coef": base_ppo.entropy_coef,
        "max_grad_norm": base_ppo.max_grad_norm,
        "adam_eps": base_ppo.adam_eps,
        "gamma": base_ppo.gamma,
        "gae_lambda": base_ppo.gae_lambda,
        "normalize_advantage": base_ppo.normalize_advantage,
        "max_episode_steps": args.max_episode_steps,
        "slack_reward": reward.slack_reward,
        "distance_scalar": reward.distance_scalar,
        "success_reward": reward.success_reward,
        "success_distance": reward.success_distance,
        "success_rule": "STOP called and geodesic distance <= success_distance",
        "nonfinite_distance_delta": -1.0,
        "checkpoints": args.checkpoints,
        "reward_window": args.reward_window,
        "resume_interval": args.resume_interval,
        "holdout": "excluded; same lexicographic tail as the imitation split",
        "navigation_benchmark": "NOT_RUN",
    }
    resolved = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    if "hydra" in resolved:
        del resolved["hydra"]
    OmegaConf.save(resolved, args.output / "config.yaml")
    (args.output / "rl_config.json").write_text(json.dumps(schedule, indent=2) + "\n", encoding="utf-8")
    log_path = args.output / "train_log.jsonl"
    window = deque(maxlen=args.reward_window)
    try:
        for update in range(args.updates):
            scaled = base_ppo.scheduled(
                update,
                args.updates,
                decay_lr=args.linear_lr_decay,
                decay_clip=args.linear_clip_decay,
            )
            for group in optimizer.param_groups:
                group["lr"] = scaled.learning_rate
            started = time.time()
            segments = []
            finished = []
            for worker in workers:
                steps, bootstrap, done_rows = worker.rollout(args.rollout_steps)
                segments.append((steps, bootstrap))
                finished.extend(done_rows)
            for row in finished:
                window.append(float(row["return"]))
            total_steps = sum(len(steps) for steps, _bootstrap in segments)
            minibatch = math.ceil(total_steps / args.mini_batches)
            stats = ppo_update(
                model,
                critic,
                optimizer,
                [],
                0.0,
                scaled,
                minibatch,
                segments,
            )
            row = {
                "update": update,
                "environments": args.environments,
                "steps": stats["steps"],
                "loss": stats["loss"],
                "policy": stats["policy"],
                "value": stats["value"],
                "entropy": stats["entropy"],
                "mean_reward": stats["mean_reward"],
                "window_return": (sum(window) / len(window)) if window else None,
                "episodes_finished": len(finished),
                "learning_rate": scaled.learning_rate,
                "clip": scaled.clip,
                "seconds": time.time() - started,
                "split": "train_scenes",
                "episodes": finished,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            print(
                f"update {update} steps {row['steps']} reward {row['mean_reward']:.4f} "
                f"policy {row['policy']:.4f} lr {row['learning_rate']:.8f}",
                flush=True,
            )
            record = {
                "update": update,
                "init": str(init_path),
                "learning_rate": scaled.learning_rate,
                "clip": scaled.clip,
            }
            due = checkpoint_due(update, args.updates, args.checkpoints)
            resume = (update + 1) % args.resume_interval == 0 or update + 1 == args.updates
            if due or resume:
                _save(args.output / "ckpt_last.pt", model, critic, optimizer, update, record)
            if due:
                _save(args.output / f"ckpt_update_{update}.pt", model, critic, optimizer, update, record)
    finally:
        for worker in workers:
            worker.close()


@hydra.main(version_base="1.3", config_path="../../configs", config_name="brain_vln_rl")
def main(cfg) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
