#!/usr/bin/env python3
"""Pure offline imitation on the collected R2R-CE train split.

The loss is uniform masked expert-action cross-entropy. Delays stay fixed.
BrainVLN_v3 additionally applies NeuCube-style online STDP to excitatory
synaptic magnitudes during the forward pass. BrainVLN_v5 is a separate Hydra
entry, ``scripts/brain_vln/train_v5.py``. This script does not run DAgger,
learnable delays, FlyWire, PPO, or a closed-loop SR/SPL evaluation. The
holdout score is teacher-forcing loss on an internal scene split of the same
collection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from spikingnav.brain_vln.batch_parallel import BatchParallel
from spikingnav.brain_vln.config import BrainVLNConfig, PopulationSizes
from spikingnav.brain_vln.data.collate import collate_fn
from spikingnav.brain_vln.data.dataset import BrainVLNDataset
from spikingnav.brain_vln.model import BrainVLNModel
from spikingnav.brain_vln.train_loop import load_checkpoint, save_checkpoint, train_batch


def _git(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return f"unavailable: {exc}"
    text = (result.stdout or result.stderr).strip()
    return text or f"exit {result.returncode}"


def _source_hash() -> str:
    digest = hashlib.sha256()
    roots = [Path("spikingnav/brain_vln"), Path("scripts/brain_vln/train_il.py")]
    files = []
    for root in roots:
        if root.is_file():
            files.append(root)
        else:
            files.extend(path for path in root.rglob("*.py") if path.is_file())
    for path in sorted(files):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _model_name(args) -> str:
    if args.brain == "v5":
        return "BrainVLN_v5"
    if args.encoder == "metaspikeformer":
        return "BrainVLN_v4"
    if args.stdp:
        return "BrainVLN_v3"
    if args.layout == "cells":
        return "BrainVLN_v2"
    return "BrainVLN"


def _config(args) -> BrainVLNConfig:
    cells = args.layout == "cells"
    spatial = 800 if cells else 512
    encoder = args.encoder
    if args.brain == "v5":
        if not cells or encoder != "mlanet" or args.stdp or args.variant != "B4":
            raise ValueError("BrainVLN_v5 is the MLANet cells B4 reservoir without STDP")
        source = (
            "BrainVLN_v5. Frozen MLANet cell layout, same counts as BrainVLN_v2 "
            "(Spatial=800, Goal=512, Episodic=1280, Action=512). Episodic neurons use "
            f"a slow leak of tau={args.episodic_tau} environment steps and receive the "
            "800 spatial cells when the view changes. Goal keeps the pooled sentence "
            "in its first 256 neurons and a pointer over the 12 sub-instruction "
            "vectors, read from the episodic membrane, in the other 256. The action "
            "readout sees Action, Goal, and Episodic membranes. A separate slow "
            f"population of {args.stop_neurons} neurons reads only Episodic and Goal "
            "and adds its output to the STOP logit. Delays and excitatory/inhibitory "
            "signs stay fixed. Encoders stay frozen. No GRU. No STDP."
        )
    elif args.stdp:
        if not cells:
            raise ValueError("BrainVLN_v3 STDP training uses the cells layout")
        if encoder != "mlanet":
            raise ValueError("BrainVLN_v3 STDP training keeps the MLANet encoder")
        source = (
            "BrainVLN_v3. Same architecture as BrainVLN_v2 (frozen MLANet, RGB 3x3 "
            "and depth 4x4 cell currents, Spatial=800, Goal=512, Episodic=1280, "
            "Action=512). Synaptic magnitudes and 1x1 convolutions are trained with "
            "the action CE loss. In addition, NeuCube-style online STDP "
            f"(a_pos={args.stdp_a_pos}, a_neg={args.stdp_a_neg}, tau={args.stdp_tau}) "
            "updates excitatory synaptic magnitudes on every internal LIF step. "
            "Delays and excitatory/inhibitory signs stay fixed. Encoders stay frozen. "
            "No GRU."
        )
    elif encoder == "metaspikeformer":
        if not cells:
            raise ValueError("BrainVLN_v4 uses the cells layout")
        source = (
            "BrainVLN_v4. Same B4 cells layout as BrainVLN_v2 (Spatial=800, Goal=512, "
            "Episodic=1280, Action=512), but RGB is a frozen Meta-SpikeFormer trunk "
            "(Spike-driven Transformer V2 / Spike2Former open backbone, T=4) pooled "
            "to a 3x3xC map. Depth stays the MLANet DD-PPO ResNet-50 4x4x192 map. "
            "Language stays the MLANet instruction LSTM + CLIP sub-instruction "
            "512-d Goal current. CLIP's visual tower and MLANet's navigation heads "
            "are released after those weights load. Trainable 1x1 convolutions map RGB/depth cells to "
            "32 channels; LayerNorm is over those 32 channels at each cell, then "
            "scaled by 0.5. Previous action is a tanh current of gain 0.25. Synaptic "
            "magnitudes and the 1x1 convolutions are trained; delays and "
            "excitatory/inhibitory signs stay fixed. Encoders stay frozen. No GRU. "
            "No STDP."
        )
    elif cells:
        source = (
            "BrainVLN_v2. Frozen MLANet encoders. RGB keeps the CLIP 3x3x768 map "
            "and depth keeps the ResNet 4x4x192 map. A trainable 1x1 convolution "
            "maps every cell to 32 channels. LayerNorm is over those 32 channels "
            "at each cell, then scaled by 0.5. Neuron order is RGB cells (288) "
            "then depth cells (512), row, column, channel, for 800 Spatial neurons. "
            "Language stays a 512-d current into 512 Goal neurons. Previous action "
            "is a tanh current of gain 0.25 added after the cell normalization. "
            "Episodic=1280, Action=512. Spatial and Episodic send spikes into Goal. "
            "Synaptic magnitudes and the 1x1 convolutions are trained; delays and "
            "excitatory/inhibitory signs stay fixed. Encoders stay frozen. No GRU."
        )
    else:
        source = (
            "frozen MLANet encoders from data/pre_trained_model/MLANet/mla_best_ppo.pth: "
            "CLIP ViT-B/32 RGB and DD-PPO ResNet-50 depth stay 256+256 and drive "
            "512 Spatial neurons; instruction LSTM and CLIP sub-instructions stay "
            "256+256 and drive 512 Goal neurons. Each half is LayerNormed without "
            "an affine, then scaled by 0.5. Previous action is a tanh current of "
            "gain 0.25 added after that normalization. Episodic=1280, Action=512. "
            "Spatial and Episodic both send spikes into Goal. Synaptic magnitudes "
            "are trained with the action loss; delays and excitatory/inhibitory "
            "signs stay fixed. Encoders stay frozen. No GRU."
        )
    return BrainVLNConfig(
        populations=PopulationSizes(spatial=spatial, goal=512, episodic=1280, action=512),
        variant=args.variant,
        out_degree=48,
        feature_dim=64,
        vocab_size=8192,
        encoder=encoder,
        encoder_root="data/pre_trained_model",
        visual_layout="cells" if cells else "pooled",
        seed=args.seed,
        event_mode="surprise",
        trainable_synapses=True,
        use_stdp=bool(args.stdp),
        stdp_a_pos=args.stdp_a_pos,
        stdp_a_neg=args.stdp_a_neg,
        stdp_tau=args.stdp_tau,
        stdp_excitatory_only=True,
        brain=args.brain,
        episodic_tau=args.episodic_tau,
        stop_neurons=args.stop_neurons,
        view_change_threshold=args.view_change_threshold,
        progress_weight=0.0,
        prediction_weight=0.0,
        event_weight=0.0,
        activity_weight=0.0,
        covariance_weight=0.0,
        weight_source=source,
    )


def _move(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _forward(model: BrainVLNModel, batch: dict):
    return model(
        batch["rgb"],
        batch["depth"],
        batch["depth_valid"],
        batch["tokens"],
        batch["token_mask"],
        batch["valid_mask"],
        batch["reset_mask"],
        batch["detach_mask"],
        actions=batch["actions"],
        instruction_text=batch.get("instruction_text"),
    )


@torch.no_grad()
def evaluate(model: BrainVLNModel, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), device=device)
    total_count = torch.zeros((), device=device)
    batches = 0
    for batch in loader:
        batch = _move(batch, device)
        logits, _state, _aux = _forward(model, batch)
        labels = batch["actions"].reshape(-1).long().clamp(min=0)
        mask = batch["loss_mask"].reshape(-1).to(logits.dtype)
        per_step = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels, reduction="none")
        total_loss = total_loss + (per_step * mask).sum()
        chosen = logits.argmax(dim=-1).reshape(-1)
        total_correct = total_correct + ((chosen == labels).to(logits.dtype) * mask).sum()
        total_count = total_count + mask.sum()
        batches += 1
    count = float(total_count.detach())
    model.train()
    return {
        "loss": float(total_loss.detach() / max(count, 1.0)),
        "accuracy": float(total_correct.detach() / max(count, 1.0)),
        "supervised_steps": int(count),
        "batches": batches,
    }


def train(args) -> None:
    output = Path(args.output)
    if output.exists() and not args.resume:
        raise SystemExit(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    root = Path(args.root)
    train_set = BrainVLNDataset(root, split="train", mode="window")
    holdout_set = BrainVLNDataset(root, split="holdout", mode="window")
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        generator=generator,
        persistent_workers=args.workers > 0,
    )
    holdout_loader = DataLoader(
        holdout_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    config = _config(args)
    model = BrainVLNModel(config).to(device)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
    )
    start_epoch = 0
    global_step = 0
    if args.resume:
        payload = load_checkpoint(Path(args.resume), model, optimizer)
        global_step = int(payload["step"]) + 1
        record = payload.get("record") or {}
        saved_epoch = int(record.get("epoch", 0))
        # An epoch summary is written only after that epoch finishes.
        start_epoch = saved_epoch + 1 if "optimizer_steps" in record else saved_epoch
        model.to(device)

    runner = model
    if args.data_parallel:
        if device.type != "cuda" or torch.cuda.device_count() < 2:
            raise SystemExit("data_parallel needs at least two visible CUDA devices")
        runner = BatchParallel(model, list(range(torch.cuda.device_count())))

    excluded = ["dagger", "learnable_delay", "flywire", "ppo", "inflection_weighted_ce"]
    if not args.stdp:
        excluded.insert(1, "stdp")
    provenance = {
        "model_name": _model_name(args),
        "encoder": args.encoder,
        "visual_layout": args.layout,
        "microbatch": args.microbatch,
        "data_parallel": bool(args.data_parallel),
        "visible_cuda_devices": torch.cuda.device_count() if device.type == "cuda" else 0,
        "algorithm": "offline imitation learning",
        "loss": "uniform masked expert-action cross-entropy",
        "stdp": bool(args.stdp),
        "stdp_reference": "NeuCube-Py STDP (Kasabov 2014 / KEDRI-AUT/NeuCube-Py)",
        "excluded": excluded,
        "navigation_metrics": "NOT_RUN",
        "holdout_meaning": "teacher-forcing loss on an internal scene split, not official Val-Seen or Val-Unseen",
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "head": _git(["rev-parse", "HEAD"]),
        "status_short": _git(["status", "--short"]),
        "source_sha256": _source_hash(),
        "command": sys.argv,
        "seed": args.seed,
        "config": config.to_dict(),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "synapse_statistics": None if model.graph is None else model.graph.statistics(),
        "synapses_trainable": bool(config.trainable_synapses),
        "input_checkpoint": args.resume or None,
        "data_root": str(root),
        "train_scenes": list(train_set.assignment["train"]),
        "holdout_scenes": list(train_set.assignment["holdout"]),
        "train_windows": len(train_set),
        "holdout_windows": len(holdout_set),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    config_file = getattr(args, "config_file", None)
    if config_file:
        provenance["config_file"] = config_file
        resolved_yaml = getattr(args, "resolved_yaml", None)
        if resolved_yaml:
            (output / "config.yaml").write_text(resolved_yaml, encoding="utf-8")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    log_path = output / "train_log.jsonl"
    started = time.time()
    history_epoch = []
    for epoch in range(start_epoch, args.epochs):
        runner.train()
        epoch_loss = 0.0
        epoch_correct = 0.0
        epoch_count = 0
        for batch in train_loader:
            batch = _move(batch, device)
            _state, parts = train_batch(
                runner,
                batch,
                optimizer,
                grad_clip=args.grad_clip,
                state=None,
                microbatch=args.microbatch,
            )
            parts["step"] = global_step
            parts["epoch"] = epoch
            epoch_loss += parts["loss"] * parts["supervised_steps"]
            epoch_correct += parts["accuracy"] * parts["supervised_steps"]
            epoch_count += parts["supervised_steps"]
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(parts) + "\n")
            if global_step % 20 == 0:
                print(json.dumps(parts), flush=True)
            if args.checkpoint_every > 0 and global_step % args.checkpoint_every == 0:
                save_checkpoint(output / "ckpt_last.pt", model, optimizer, global_step, parts)
            global_step += 1
        epoch_row = {
            "epoch": epoch,
            "loss": epoch_loss / max(epoch_count, 1),
            "accuracy": epoch_correct / max(epoch_count, 1),
            "supervised_steps": epoch_count,
            "optimizer_steps": global_step,
        }
        history_epoch.append(epoch_row)
        save_checkpoint(
            output / f"ckpt_epoch_{epoch}.pt",
            model,
            optimizer,
            global_step - 1,
            epoch_row,
        )
        save_checkpoint(output / "ckpt_last.pt", model, optimizer, global_step - 1, epoch_row)
        print(json.dumps({"epoch_done": epoch_row}), flush=True)

    holdout = evaluate(runner, holdout_loader, device)
    summary = {
        "status": "completed",
        "variant": args.variant,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "train_windows": len(train_set),
        "holdout_windows": len(holdout_set),
        "epoch_metrics": history_epoch,
        "holdout_teacher_forcing": holdout,
        "sr": "NOT_RUN",
        "spl": "NOT_RUN",
        "ne": "NOT_RUN",
        "ndtw": "NOT_RUN",
        "elapsed_s": time.time() - started,
        "weight_source": config.weight_source,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary}), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="storage/collect_data/train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", default="B4", choices=["B0", "B1", "B2", "B3", "B4", "B5"])
    parser.add_argument("--layout", default="pooled", choices=["pooled", "cells"])
    parser.add_argument(
        "--encoder",
        default="mlanet",
        choices=["mlanet", "metaspikeformer"],
        help="BrainVLN_v4 uses metaspikeformer (RGB) while keeping MLANet depth/language",
    )
    parser.add_argument(
        "--v5",
        action="store_true",
        help="BrainVLN_v5 only. The v2 default is unchanged; prefer scripts/brain_vln/train_v5.py",
    )
    parser.add_argument("--episodic-tau", type=float, default=30.0)
    parser.add_argument("--stop-neurons", type=int, default=64)
    parser.add_argument("--view-change-threshold", type=float, default=0.05)
    parser.add_argument(
        "--stdp",
        action="store_true",
        help="BrainVLN_v3: enable NeuCube-style online STDP (requires --layout cells)",
    )
    parser.add_argument("--stdp-a-pos", type=float, default=1e-4)
    parser.add_argument("--stdp-a-neg", type=float, default=1e-2)
    parser.add_argument("--stdp-tau", type=float, default=3.0)
    parser.add_argument("--microbatch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--data-parallel",
        action="store_true",
        help="Split each microbatch across every visible CUDA device",
    )
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--resume", default="")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.brain = "v5" if args.v5 else "v2"
    if args.epochs < 1 or args.batch_size < 1 or args.microbatch < 1:
        parser.error("epochs, batch-size, and microbatch must be positive")
    if args.stdp and args.layout != "cells":
        parser.error("BrainVLN_v3 (--stdp) requires --layout cells")
    if args.encoder == "metaspikeformer" and args.layout != "cells":
        parser.error("BrainVLN_v4 (--encoder metaspikeformer) requires --layout cells")
    if args.encoder == "metaspikeformer" and args.stdp:
        parser.error("BrainVLN_v4 does not enable STDP; omit --stdp")
    if args.v5 and (args.stdp or args.layout != "cells" or args.encoder != "mlanet" or args.variant != "B4"):
        parser.error("BrainVLN_v5 requires --layout cells --variant B4 --encoder mlanet, without --stdp")
    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
