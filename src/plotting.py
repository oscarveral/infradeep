"""Training metrics tracker and matplotlib plot generation.

After each epoch the training scripts call ``TrainingTracker.save_plots()``,
which writes a four-panel PNG to ``output_dir``:

1. Training loss per logged step (fine-grained progress).
2. Train / val loss per epoch (convergence overview).
3. Learning-rate curve per step (Noam or linear schedule).
4. Tokens / sec per epoch (throughput benchmark).

A JSON file with all raw numbers is also written alongside the PNG.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works without a display
import matplotlib.pyplot as plt


class TrainingTracker:
    """Accumulates training metrics and saves plots after every epoch.

    Args:
        output_dir (str): Directory where the PNG and JSON are written.
            Created automatically if it does not exist.
        run_name (str, optional): Prefix for output file names.
            Defaults to ``"run"``.
    """

    def __init__(self, output_dir: str, run_name: str = "run") -> None:
        self.output_dir = output_dir
        self.run_name = run_name
        os.makedirs(output_dir, exist_ok=True)

        # Fine-grained (per log-interval step)
        self.steps: list[int] = []
        self.step_losses: list[float] = []
        self.step_lrs: list[float] = []

        # Coarse (per epoch)
        self.epochs: list[int] = []
        self.epoch_train_losses: list[float] = []
        self.epoch_val_losses: list[float] = []
        self.epoch_bleu: list[Optional[float]] = []
        self.epoch_tok_per_sec: list[float] = []

    def log_step(self, step: int, loss: float, lr: float) -> None:
        """Record a single training step's loss and learning rate.

        Args:
            step (int): Global training step number.
            loss (float): Batch cross-entropy loss.
            lr (float): Current learning rate.
        """
        self.steps.append(step)
        self.step_losses.append(loss)
        self.step_lrs.append(lr)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        tokens_per_sec: float,
        bleu: Optional[float] = None,
    ) -> None:
        """Record epoch-level metrics.

        Args:
            epoch (int): Epoch number (1-indexed).
            train_loss (float): Average training loss for the epoch.
            val_loss (float): Validation loss after the epoch.
            tokens_per_sec (float): Average throughput (tokens/sec).
            bleu (float, optional): BLEU score if computed this epoch.
        """
        self.epochs.append(epoch)
        self.epoch_train_losses.append(train_loss)
        self.epoch_val_losses.append(val_loss)
        self.epoch_tok_per_sec.append(tokens_per_sec)
        self.epoch_bleu.append(bleu)

    def save_plots(self) -> None:
        """Write a four-panel training curve PNG and a JSON metrics file.

        The PNG is overwritten after each call, so the latest epoch's
        state is always available in the output directory.

        Panels:
            - Top-left:  Training loss per step (fine-grained).
            - Top-right: Train + val loss per epoch.
            - Bottom-left: Learning rate per step.
            - Bottom-right: BLEU per epoch (if any) or tokens/sec per epoch.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Training curves — {self.run_name}", fontsize=13)

        # ── Panel 1: step-level train loss ────────────────────────────────
        ax = axes[0, 0]
        if self.steps:
            ax.plot(self.steps, self.step_losses, linewidth=0.8, color="steelblue", alpha=0.8)
            # Smooth with a simple running average for readability
            if len(self.step_losses) >= 20:
                w = max(1, len(self.step_losses) // 50)
                smoothed = _smooth(self.step_losses, w)
                ax.plot(self.steps, smoothed, linewidth=1.8, color="navy", label="smoothed")
                ax.legend(fontsize=8)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Train loss (per step)")
        ax.grid(True, alpha=0.3)

        # ── Panel 2: epoch-level train + val loss ─────────────────────────
        ax = axes[0, 1]
        if self.epochs:
            ax.plot(self.epochs, self.epoch_train_losses, marker="o", label="Train", color="steelblue")
            ax.plot(self.epochs, self.epoch_val_losses, marker="s", label="Val", color="tomato")
            ax.legend(fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Train / Val loss (per epoch)")
        ax.grid(True, alpha=0.3)

        # ── Panel 3: learning rate curve ──────────────────────────────────
        ax = axes[1, 0]
        if self.steps:
            ax.plot(self.steps, self.step_lrs, linewidth=1.2, color="darkorange")
        ax.set_xlabel("Step")
        ax.set_ylabel("LR")
        ax.set_title("Learning rate schedule")
        ax.grid(True, alpha=0.3)
        # Scientific notation on y-axis
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        # ── Panel 4: BLEU (if available) else tokens/sec ──────────────────
        ax = axes[1, 1]
        bleu_values = [b for b in self.epoch_bleu if b is not None]
        bleu_epochs = [e for e, b in zip(self.epochs, self.epoch_bleu) if b is not None]
        if bleu_values:
            ax.plot(bleu_epochs, bleu_values, marker="^", color="mediumseagreen")
            ax.set_ylabel("BLEU")
            ax.set_title("BLEU score (greedy, first 2k val)")
        elif self.epoch_tok_per_sec:
            ax.plot(self.epochs, self.epoch_tok_per_sec, marker="D", color="mediumpurple")
            ax.set_ylabel("Tokens / sec")
            ax.set_title("Throughput (tokens / sec)")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        png_path = os.path.join(self.output_dir, f"{self.run_name}_training_curves.png")
        fig.savefig(png_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        self._save_json()

    def _save_json(self) -> None:
        """Persist all accumulated metrics to a JSON file.

        The file is written to ``output_dir/{run_name}_metrics.json``.
        """
        data = {
            "steps": self.steps,
            "step_losses": self.step_losses,
            "step_lrs": self.step_lrs,
            "epochs": self.epochs,
            "epoch_train_losses": self.epoch_train_losses,
            "epoch_val_losses": self.epoch_val_losses,
            "epoch_bleu": self.epoch_bleu,
            "epoch_tok_per_sec": self.epoch_tok_per_sec,
        }
        json_path = os.path.join(self.output_dir, f"{self.run_name}_metrics.json")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)


def _smooth(values: list[float], window: int) -> list[float]:
    """Apply a simple uniform moving average.

    Args:
        values (list[float]): Input values.
        window (int): Smoothing window size.

    Returns:
        list[float]: Smoothed values, same length as input.
    """
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window // 2)
        end = min(len(values), i + window // 2 + 1)
        smoothed.append(sum(values[start:end]) / (end - start))
    return smoothed
