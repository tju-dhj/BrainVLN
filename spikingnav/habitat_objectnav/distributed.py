"""One process per GPU for Habitat PPO and imitation.

Each rank owns its own simulators and recurrent state. After backward, gradients
are averaged once. The policy is not wrapped in DistributedDataParallel: the
update replays the rollout step by step, and that repeated forward trips DDP's
single-forward hook.
"""

from __future__ import annotations

import os

import torch


def preemption_triggered(finished: int, world_size: int, threshold: float) -> bool:
    """True once ``threshold`` of the workers have finished the full rollout.

    With two workers, 0.6 never fires (1/2 = 0.5). Use 0.5 so the faster
    worker can cut the slower one off. With four workers, 0.6 fires at 3.
    """
    if world_size <= 1 or threshold <= 0.0:
        return False
    return finished >= threshold * world_size


def distributed_context() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def rollout_steps_global(rollout_steps: int, num_envs: int, world_size: int) -> int:
    return int(rollout_steps) * int(num_envs) * int(world_size)


def _active() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def init_distributed(local_rank: int, world_size: int) -> None:
    if world_size <= 1:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("distributed training needs CUDA")
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")


def shutdown_distributed() -> None:
    if _active():
        torch.distributed.destroy_process_group()


def broadcast_parameters(model: torch.nn.Module) -> None:
    if not _active():
        return
    for tensor in list(model.parameters()) + list(model.buffers()):
        torch.distributed.broadcast(tensor.data, src=0)


def average_gradients(model: torch.nn.Module, sample_count: float = None) -> None:
    """Average gradients across workers.

    ``sample_count`` weights each worker by how many environment steps it
    actually kept. Equal counts match a plain mean over workers. Shorter
    preempted rollouts then contribute proportionally less.
    """
    if not _active():
        return
    grads = [param.grad for param in model.parameters() if param.grad is not None]
    if not grads:
        return
    flat = torch.cat([grad.detach().reshape(-1) for grad in grads])
    if sample_count is None:
        torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
        flat.div_(torch.distributed.get_world_size())
    else:
        weight = flat.new_tensor([float(sample_count)])
        flat = torch.cat([flat * weight, weight])
        torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
        total = flat[-1].clamp(min=1.0)
        flat = flat[:-1] / total
    offset = 0
    for grad in grads:
        numel = grad.numel()
        grad.copy_(flat[offset : offset + numel].view_as(grad))
        offset += numel


class RolloutPreemption:
    """DD-PPO straggler preemption (Wijmans et al., 2020).

    A worker that finishes all ``rollout_steps`` publishes that fact. Every
    other worker checks between simulator steps. Once the finished fraction
    reaches ``threshold``, a worker that already has ``min_fraction`` of its
    rollout stops and joins the update. Workers that were not cut short still
    contribute a full rollout.
    """

    def __init__(self, threshold: float = 0.6, min_fraction: float = 0.5) -> None:
        self.threshold = float(threshold)
        self.min_fraction = float(min_fraction)

    def _enabled(self) -> bool:
        return _active() and self.threshold > 0.0

    def _store(self):
        return torch.distributed.distributed_c10d._get_default_store()

    @staticmethod
    def _key(update_idx: int) -> str:
        return f"ddppo-finished-{int(update_idx)}"

    def should_stop(self, update_idx: int, steps_done: int, rollout_steps: int) -> bool:
        if not self._enabled():
            return False
        earliest = max(1, int(rollout_steps * self.min_fraction))
        if steps_done < earliest:
            return False
        finished = int(self._store().add(self._key(update_idx), 0))
        return preemption_triggered(
            finished, torch.distributed.get_world_size(), self.threshold
        )

    def mark_finished(self, update_idx: int) -> None:
        if not self._enabled():
            return
        self._store().add(self._key(update_idx), 1)

    def barrier(self) -> None:
        if _active():
            torch.distributed.barrier()


def reduce_sum(value: float, device: torch.device) -> float:
    if not _active():
        return float(value)
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return float(tensor.item())


def reduce_mean(value: float, device: torch.device) -> float:
    if not _active():
        return float(value)
    return reduce_sum(value, device) / torch.distributed.get_world_size()


def normalize_distributed(values: torch.Tensor) -> torch.Tensor:
    """Match ``Tensor.std`` (unbiased) on one GPU, and on the joined batch when distributed."""
    if not _active():
        return (values - values.mean()) / (values.std() + 1e-8)
    total = values.sum()
    sq = values.square().sum()
    count = torch.tensor(float(values.numel()), device=values.device, dtype=values.dtype)
    packed = torch.stack([total, sq, count])
    torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
    count_g = packed[2].clamp(min=1)
    mean = packed[0] / count_g
    var = (packed[1] - packed[0].square() / count_g) / (count_g - 1).clamp(min=1)
    std = var.clamp(min=0).sqrt()
    return (values - mean) / (std + 1e-8)
