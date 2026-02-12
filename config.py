"""
Configuration for KV Cache Ghost Training v5
Input: hidden_states.pt + kv_cache_reuse.pt → kv_cache_no_reuse.pt
"""
from dataclasses import dataclass, field
from typing import List
import os


@dataclass
class TrainingConfig:
    # Data paths
    data_root: str = r"/path/to/data"  # 修改为实际数据路径
    task_dirs: List[str] = field(default_factory=lambda: [f"task_{i}" for i in range(3)])

    # Model — per-layer independent, accuracy first
    hidden_state_dim: int = 4096   # hidden_states 输入维度
    embed_dim: int = 256           # hidden state encoder 输出维度
    reuse_proj_dim: int = 256      # reuse_kv 投影维度
    hidden_dim: int = 1024         # MLP 隐藏层维度
    num_layers: int = 32
    num_heads: int = 32
    kv_dim: int = 128

    # KV cache indexing (图像 patch 索引 1-257)
    kv_seq_start: int = 1
    kv_seq_end: int = 257          # 共256个

    # Training
    batch_size: int = 2            # per GPU (per-layer独立模型占显存更多)
    num_epochs: int = 100
    base_lr: float = 3e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # Learning rate schedule
    warmup_steps: int = 1000

    # Data split
    val_ratio: float = 0.1

    # Dataloader — 大量 worker 并行加载饱和磁盘带宽
    num_workers: int = 16          # 每 GPU 16 workers, 总 128 → 饱和 128 核
    prefetch_factor: int = 4       # 每 worker 预取 4 batch

    # Checkpointing
    checkpoint_dir: str = r"./checkpoints"
    save_every_steps: int = 1000
    val_every_steps: int = 500
    log_every_steps: int = 10

    # Mixed precision
    use_amp: bool = True

    # Visualization
    loss_plot_dir: str = r"./plots"
    moving_avg_window: int = 50    # train loss moving average 窗口

    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.loss_plot_dir, exist_ok=True)
