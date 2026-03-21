"""Shared utility functions for training and evaluation.

Standalone helpers used by both training scripts. No cross-module dependencies
beyond PyTorch and the standard library, so this module can be imported first
in any script.

Contents:
    create_logger: Configures Python logging for console and optional file output.
    count_parameters: Prints model parameter counts and memory estimate.
    save_checkpoint: Serialises full training state to disk.
    load_checkpoint: Restores training state from a checkpoint file.
    compute_bleu: SacreBLEU corpus-level BLEU evaluation.
    load_yaml_config: Loads a YAML configuration file.
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Optional

import sacrebleu
import torch
import torch.nn as nn
from torch.optim import Optimizer


def create_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create and configure a named Python logger.

    Attaches a StreamHandler (always) and an optional FileHandler. Guards
    against duplicate handlers if the same logger name is requested twice.

    Args:
        name (str): Logger name, typically ``__name__`` of the calling module.
        log_file (str, optional): Path to a ``.log`` file. When provided,
            messages are written to disk in addition to the console.
        level (int, optional): Logging level. Defaults to ``logging.INFO``.

    Returns:
        logging.Logger: Configured logger instance.

    Example:
        >>> logger = create_logger(__name__, log_file="logs/train.log")
        >>> logger.info("Epoch 1 started")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def count_parameters(model: nn.Module) -> int:
    """Print a breakdown of model parameter counts and memory footprint.

    Reports total, trainable, and non-trainable parameter counts together with
    approximate weight memory for fp32 and fp16. The memory breakdown is
    educational: at fp32 a 220 M-param model occupies ~840 MB for weights alone.
    AdamW adds first- and second-moment tensors (~1.68 GB more), and activations
    during the forward pass add several additional gigabytes. Total training
    memory is typically 3–4× weight size, which motivates hardware-optimisation
    techniques such as mixed precision, ZeRO, and FSDP.

    Args:
        model (nn.Module): Any PyTorch model.

    Returns:
        int: Total number of parameters.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable

    print("=" * 55)
    print(f"  Model parameter summary")
    print("=" * 55)
    print(f"  Total parameters      : {total:>15,}")
    print(f"  Trainable parameters  : {trainable:>15,}")
    print(f"  Non-trainable params  : {non_trainable:>15,}")
    print("-" * 55)
    fp32_mb = total * 4 / (1024 ** 2)
    fp16_mb = total * 2 / (1024 ** 2)
    print(f"  Weight memory (fp32)  : {fp32_mb:>12.1f} MB")
    print(f"  Weight memory (fp16)  : {fp16_mb:>12.1f} MB  ← after mixed precision")
    print(f"  Adam state (fp32)     : {fp32_mb * 2:>12.1f} MB  ← 2× weight size")
    print("=" * 55)

    return total


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    step: int,
    loss: float,
    path: str,
    cfg: dict,
    keep_last_n: int = 3,
) -> None:
    """Save a full training snapshot to disk.

    The checkpoint contains everything needed to resume training exactly:
    model weights, optimiser state (Adam moments), epoch, global step, loss,
    and the config dict. Embedding the config means the architecture can be
    reconstructed from the checkpoint alone without hunting for the original YAML.

    After saving, older checkpoints in the same directory are rotated so that
    only the ``keep_last_n`` most recent files are kept, preventing disk
    exhaustion during long training runs.

    Args:
        model (nn.Module): The PyTorch model (on any device).
        optimizer (Optimizer): The optimiser whose state will be saved.
        epoch (int): Completed epoch number (1-indexed).
        step (int): Global training step at the point of saving.
        loss (float): Validation loss at this checkpoint.
        path (str): Full file path for the checkpoint
            (e.g. ``"models/ckpt_ep01.pt"``).
        cfg (dict): Training config dict embedded in the checkpoint.
        keep_last_n (int, optional): Number of most-recent checkpoints to
            retain. Set to ``0`` to disable rotation. Defaults to ``3``.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    state = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
    }
    torch.save(state, path)

    if keep_last_n > 0:
        checkpoint_dir = os.path.dirname(path)
        all_ckpts = sorted(
            glob.glob(os.path.join(checkpoint_dir, "*.pt")),
            key=os.path.getmtime,
        )
        for old_ckpt in all_ckpts[:-keep_last_n]:
            os.remove(old_ckpt)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Load a training snapshot from disk and restore model (and optionally optimiser) state.

    The ``map_location`` argument allows loading a checkpoint saved on a GPU
    onto a CPU machine, which is common when students submit cluster jobs and
    later inspect results on a laptop.

    Args:
        path (str): Path to the ``.pt`` checkpoint file.
        model (nn.Module): Model instance whose weights will be restored.
            The architecture must match the one used when saving.
        optimizer (Optimizer, optional): Optimiser to restore state into.
            Pass ``None`` for inference-only or when starting a new fine-tune.
        device (torch.device, optional): Target device for weight loading.
            Defaults to CPU.

    Returns:
        dict: Metadata with keys ``"epoch"`` (int), ``"step"`` (int),
            and ``"loss"`` (float).
    """
    state = torch.load(path, map_location=device)

    model.load_state_dict(state["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    return {
        "epoch": state["epoch"],
        "step": state["step"],
        "loss": state["loss"],
    }



def compute_bleu(
    predictions: list[str],
    references: list[list[str]],
) -> float:
    """Compute corpus-level BLEU using SacreBLEU.

    Raw BLEU scores are not reproducible across codebases because they depend
    on the tokenisation strategy used before counting n-gram matches. Different
    tokenisers produce different BLEU values for identical translations, making
    cross-paper comparison meaningless. SacreBLEU standardises tokenisation
    (using the "13a" tokeniser by default) so scores are directly comparable
    with published results.

    Args:
        predictions (list[str]): Decoded hypothesis strings, one per example.
        references (list[list[str]]): Reference lists where each hypothesis may
            have multiple gold references. Single-reference shorthand::

                [[ref] for ref in ref_list]

    Returns:
        float: Corpus BLEU score in the 0–100 range used by SacreBLEU.

    Example:
        >>> bleu = compute_bleu(["hola mundo"], [["hola mundo"]])
        >>> print(f"BLEU: {bleu:.2f}")
    """
    result = sacrebleu.corpus_bleu(predictions, references)
    return result.score


def load_yaml_config(path: str) -> dict:
    """Load a YAML configuration file and return it as a nested dict.

    Args:
        path (str): Path to the ``.yaml`` file (relative or absolute).

    Returns:
        dict: Parsed configuration.
    """
    import yaml

    with open(path, "r") as f:
        return yaml.safe_load(f)
