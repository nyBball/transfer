"""
Multi-GPU DDP Training for KV Cache Prediction v5
Input: hidden_states.pt + kv_cache_reuse.pt → kv_cache_no_reuse.pt
Per-layer independent models for maximum accuracy.

Usage: torchrun --nproc_per_node=8 train.py
"""
import os
import sys
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from config import TrainingConfig
from dataset import create_dataloaders
from model import KVCacheModel
from scheduler import create_scheduler
from visualize import LossVisualizer


def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank = 0
        local_rank = 0
        world_size = 1

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


@torch.no_grad()
def validate(model, val_loader, device):
    """Run validation and return average loss across all GPUs."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in val_loader:
        hidden_states = batch["hidden_states"].to(device, non_blocking=True)
        reuse_kv = batch["reuse_kv"].to(device, non_blocking=True)
        target_kv = batch["target_kv"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(hidden_states, reuse_kv, target_kv)
            loss = output["loss"]

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(1, num_batches)

    if dist.is_initialized():
        loss_tensor = torch.tensor([avg_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()

    model.train()
    return avg_loss


def save_checkpoint(model, optimizer, scheduler, scaler, step, loss, path):
    """Save training checkpoint."""
    model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    checkpoint = {
        "step": step,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, path)


def train(config: TrainingConfig):
    """Main training function."""
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0

    if is_main:
        print(f"[Main] Training with {world_size} GPUs")
        print(f"[Main] batch_size={config.batch_size}/GPU, lr={config.base_lr}")
        print(f"[Main] Per-layer independent models (no weight sharing)")

    # Dataloaders
    train_loader, val_loader, num_train = create_dataloaders(
        config, rank=rank, world_size=world_size
    )

    # Steps calculation
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * config.num_epochs

    if is_main:
        print(f"[Main] Steps/epoch: {steps_per_epoch}, Total: {total_steps}")

    # Model
    model = KVCacheModel(
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        kv_dim=config.kv_dim,
        hidden_state_dim=config.hidden_state_dim,
        embed_dim=config.embed_dim,
        reuse_proj_dim=config.reuse_proj_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)

    if is_main:
        print(f"[Main] Model parameters: {model.count_parameters():,}")

    # DDP wrapper
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.base_lr,
        weight_decay=config.weight_decay,
    )

    # Scheduler
    scheduler = create_scheduler(optimizer, config, total_steps)

    # Mixed precision scaler (using new API)
    scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp)

    # Visualizer
    if is_main:
        visualizer = LossVisualizer(
            save_dir=config.loss_plot_dir,
            moving_avg_window=config.moving_avg_window,
        )

    # ---- Training Loop ----
    global_step = 0
    model.train()

    for epoch in range(config.num_epochs):
        train_loader.sampler.set_epoch(epoch)

        # Step-based progress bar (only main rank)
        if is_main:
            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{config.num_epochs}",
                total=steps_per_epoch,
                unit="step",
            )
        else:
            pbar = train_loader

        for batch in pbar:
            # Move data to GPU (non-blocking for overlap)
            hidden_states = batch["hidden_states"].to(device, non_blocking=True)
            reuse_kv = batch["reuse_kv"].to(device, non_blocking=True)
            target_kv = batch["target_kv"].to(device, non_blocking=True)

            # Forward with AMP (bfloat16 autocast)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=config.use_amp):
                output = model(hidden_states, reuse_kv, target_kv)
                loss = output["loss"]

            # Backward
            scaler.scale(loss).backward()

            # Gradient clipping
            if config.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1

            # ---- Logging ----
            if is_main and global_step % config.log_every_steps == 0:
                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix(
                    loss=f"{loss.item():.6f}",
                    lr=f"{current_lr:.2e}",
                )
                visualizer.add_train_loss(global_step, loss.item())

            # ---- Validation ----
            if global_step % config.val_every_steps == 0:
                val_loss = validate(model, val_loader, device)
                if is_main:
                    print(f"\n[Step {global_step}] Val Loss: {val_loss:.6f}")
                    visualizer.add_val_loss(global_step, val_loss)
                    visualizer.save_plot()

            # ---- Checkpoint ----
            if is_main and global_step % config.save_every_steps == 0:
                ckpt_path = Path(config.checkpoint_dir) / f"step_{global_step}.pt"
                save_checkpoint(
                    model, optimizer, scheduler, scaler, global_step, loss.item(), ckpt_path
                )
                print(f"\n[Step {global_step}] Checkpoint -> {ckpt_path}")

    # ---- Final save ----
    if is_main:
        ckpt_path = Path(config.checkpoint_dir) / "final.pt"
        save_checkpoint(
            model, optimizer, scheduler, scaler, global_step, loss.item(), ckpt_path
        )
        visualizer.save_plot()
        visualizer.save_data()
        print(f"[Main] Training complete! Final checkpoint: {ckpt_path}")

    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description="KV Cache Training v5")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    config = TrainingConfig()

    if args.data_root:
        config.data_root = args.data_root
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.num_epochs = args.epochs
    if args.lr:
        config.base_lr = args.lr
    if args.warmup_steps:
        config.warmup_steps = args.warmup_steps
    if args.num_workers is not None:
        config.num_workers = args.num_workers

    train(config)


if __name__ == "__main__":
    main()
