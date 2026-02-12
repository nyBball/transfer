"""
Training Loss Visualization with Matplotlib (per-step)
- Moving average for train loss
- Train/val loss saved separately + combined plot
"""
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for server
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from typing import Optional, List


class LossVisualizer:
    """
    Accumulate per-step train/val losses and save matplotlib plots.
    Train loss includes moving average smoothing.
    Train and val loss are saved to separate files.
    """

    def __init__(self, save_dir: str = "./plots", moving_avg_window: int = 50):
        self.save_dir = save_dir
        self.moving_avg_window = moving_avg_window

        self.train_steps: List[int] = []
        self.train_losses: List[float] = []
        self.val_steps: List[int] = []
        self.val_losses: List[float] = []

        os.makedirs(save_dir, exist_ok=True)

    def add_train_loss(self, step: int, loss: float):
        self.train_steps.append(step)
        self.train_losses.append(loss)

    def add_val_loss(self, step: int, loss: float):
        self.val_steps.append(step)
        self.val_losses.append(loss)

    @staticmethod
    def _moving_average(values: List[float], window: int) -> np.ndarray:
        """Compute moving average with the given window size."""
        if len(values) < window:
            window = max(1, len(values))
        cumsum = np.cumsum(np.insert(np.array(values, dtype=np.float64), 0, 0))
        ma = (cumsum[window:] - cumsum[:-window]) / window
        # Pad the beginning with partial averages
        prefix = [np.mean(values[:i+1]) for i in range(min(window - 1, len(values)))]
        return np.concatenate([prefix, ma])

    def save_plot(self):
        """Save separate train loss, val loss, and combined plots."""
        self._save_train_plot()
        self._save_val_plot()
        self._save_combined_plot()

    def _save_train_plot(self):
        """Save train loss plot with raw + moving average."""
        if not self.train_steps:
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        # Raw train loss (light, transparent)
        ax.plot(
            self.train_steps, self.train_losses,
            label="Train Loss (raw)", alpha=0.3, linewidth=0.8,
            color="#90CAF9",
        )

        # Moving average (solid)
        if len(self.train_losses) > 1:
            ma = self._moving_average(self.train_losses, self.moving_avg_window)
            ax.plot(
                self.train_steps, ma,
                label=f"Train Loss (MA-{self.moving_avg_window})",
                alpha=0.9, linewidth=2, color="#1565C0",
            )

        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(os.path.join(self.save_dir, "train_loss.png"), dpi=150)
        plt.close(fig)

    def _save_val_plot(self):
        """Save validation loss plot."""
        if not self.val_steps:
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(
            self.val_steps, self.val_losses,
            label="Val Loss", alpha=0.9, linewidth=2,
            marker="o", markersize=4, color="#FF5722",
        )

        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Validation Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(os.path.join(self.save_dir, "val_loss.png"), dpi=150)
        plt.close(fig)

    def _save_combined_plot(self):
        """Save combined train + val loss plot."""
        fig, ax = plt.subplots(figsize=(14, 7))

        if self.train_steps:
            # Raw (faded)
            ax.plot(
                self.train_steps, self.train_losses,
                alpha=0.2, linewidth=0.5, color="#90CAF9",
            )
            # Moving average
            if len(self.train_losses) > 1:
                ma = self._moving_average(self.train_losses, self.moving_avg_window)
                ax.plot(
                    self.train_steps, ma,
                    label=f"Train (MA-{self.moving_avg_window})",
                    alpha=0.9, linewidth=2, color="#1565C0",
                )

        if self.val_steps:
            ax.plot(
                self.val_steps, self.val_losses,
                label="Val Loss", alpha=0.9, linewidth=2,
                marker="o", markersize=4, color="#FF5722",
            )

        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("KV Cache Training Loss (Combined)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(os.path.join(self.save_dir, "loss_curve.png"), dpi=150)
        plt.close(fig)

    def save_data(self):
        """Save raw loss data to JSON files for later analysis."""
        train_data = {
            "steps": self.train_steps,
            "losses": self.train_losses,
        }
        val_data = {
            "steps": self.val_steps,
            "losses": self.val_losses,
        }

        with open(os.path.join(self.save_dir, "train_loss_data.json"), "w") as f:
            json.dump(train_data, f)

        with open(os.path.join(self.save_dir, "val_loss_data.json"), "w") as f:
            json.dump(val_data, f)
