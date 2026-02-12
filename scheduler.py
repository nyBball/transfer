"""
WarmupCosine Learning Rate Scheduler
"""
import math
from torch.optim.lr_scheduler import LRScheduler


class WarmupCosineScheduler(LRScheduler):
    """
    Linear warmup followed by cosine annealing to min_lr.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        base_lr: float,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1

        if step < self.warmup_steps:
            lr = self.base_lr * step / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            progress = min(1.0, progress)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )

        return [lr for _ in self.optimizer.param_groups]


def create_scheduler(optimizer, config, total_steps: int) -> WarmupCosineScheduler:
    return WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=config.warmup_steps,
        total_steps=total_steps,
        base_lr=config.base_lr,
        min_lr=config.min_lr,
    )
