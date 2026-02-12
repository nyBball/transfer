"""
Dataset for KV Cache Ghost Training v5
Input: hidden_states.pt + kv_cache_reuse.pt → kv_cache_no_reuse.pt
Strategy: on-demand disk loading + aggressive worker parallelism.

~15TB data, cannot fit in RAM.
Use 16 workers/GPU × 8 GPUs = 128 parallel loaders,
with prefetch_factor=4 to keep GPU pipeline full.
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from pathlib import Path
from typing import List, Tuple, Dict


class KVCacheDataset(Dataset):
    """
    On-demand loading dataset:
    1. Pre-scan all step paths at init (index in memory, ~few MB)
    2. Load & slice per __getitem__ (on worker threads)
    3. 16 workers/GPU saturate disk bandwidth
    """

    def __init__(
        self,
        data_root: str,
        task_dirs: List[str],
        kv_seq_start: int = 1,
        kv_seq_end: int = 257,
    ):
        self.kv_seq_start = kv_seq_start
        self.kv_seq_end = kv_seq_end

        # Pre-scan all step dirs (only paths, no data)
        self.step_paths: List[str] = []
        self._scan_directories(Path(data_root), task_dirs)
        print(f"[Dataset] Found {len(self.step_paths)} steps")

    def _scan_directories(self, data_root: Path, task_dirs: List[str]):
        for task_dir in task_dirs:
            task_path = data_root / task_dir
            if not task_path.exists():
                print(f"[Dataset] Warning: {task_path} not found, skipping")
                continue
            for episode_dir in sorted(task_path.iterdir()):
                if not episode_dir.is_dir():
                    continue
                for step_dir in sorted(episode_dir.iterdir()):
                    if not step_dir.is_dir():
                        continue
                    h = step_dir / "hidden_states.pt"
                    r = step_dir / "kv_cache_reuse.pt"
                    n = step_dir / "kv_cache_no_reuse.pt"
                    if h.exists() and r.exists() and n.exists():
                        self.step_paths.append(str(step_dir))

    def __len__(self) -> int:
        return len(self.step_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        step_path = self.step_paths[idx]
        s, e = self.kv_seq_start, self.kv_seq_end

        # ---- Load hidden_states (float32) ----
        hs_data = torch.load(
            os.path.join(step_path, "hidden_states.pt"),
            map_location="cpu", weights_only=True,
        )
        # (1, seq_len, 4096) → (256, 4096)
        hidden_states = hs_data["hidden_states"][0, s:e, :].contiguous()
        del hs_data

        # ---- Load KV caches (bfloat16) ----
        kv_reuse_raw = torch.load(
            os.path.join(step_path, "kv_cache_reuse.pt"),
            map_location="cpu", weights_only=True,
        )
        kv_noreuse_raw = torch.load(
            os.path.join(step_path, "kv_cache_no_reuse.pt"),
            map_location="cpu", weights_only=True,
        )

        # Extract & stack: slice seq dim [1:257], squeeze batch dim
        reuse_list = []
        target_list = []
        for layer_idx in range(32):
            lk = f"layer_{layer_idx}"
            # (32, seq_len, 128) → (32, 256, 128)
            rk = kv_reuse_raw[lk]["key"][0, :, s:e, :]
            rv = kv_reuse_raw[lk]["value"][0, :, s:e, :]
            tk = kv_noreuse_raw[lk]["key"][0, :, s:e, :]
            tv = kv_noreuse_raw[lk]["value"][0, :, s:e, :]
            reuse_list.append(torch.stack([rk, rv], dim=0))    # (2, 32, 256, 128)
            target_list.append(torch.stack([tk, tv], dim=0))

        reuse_kv = torch.stack(reuse_list, dim=0)   # (32, 2, 32, 256, 128)
        target_kv = torch.stack(target_list, dim=0)

        del kv_reuse_raw, kv_noreuse_raw

        return {
            "hidden_states": hidden_states,  # (256, 4096) float32
            "reuse_kv": reuse_kv,            # (32, 2, 32, 256, 128) bfloat16
            "target_kv": target_kv,          # (32, 2, 32, 256, 128) bfloat16
        }


def create_dataloaders(
    config,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[DataLoader, DataLoader, int]:
    """Create train/val dataloaders with aggressive parallelism."""
    full_dataset = KVCacheDataset(
        data_root=config.data_root,
        task_dirs=config.task_dirs,
        kv_seq_start=config.kv_seq_start,
        kv_seq_end=config.kv_seq_end,
    )

    # 90/10 split (deterministic)
    num_samples = len(full_dataset)
    num_val = int(num_samples * config.val_ratio)
    num_train = num_samples - num_val

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [num_train, num_val], generator=generator
    )

    if rank == 0:
        print(f"[Data] Train: {num_train}, Val: {num_val}")

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank,
        shuffle=True, drop_last=True,
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank,
        shuffle=False, drop_last=False,
    )

    # Aggressive parallelism: 16 workers/GPU, prefetch 4 batches ahead
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        sampler=val_sampler,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_loader, val_loader, num_train
