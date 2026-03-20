"""Fine-tuning script for Option B (pre-trained MarianMT).

Fine-tunes ``Helsinki-NLP/opus-mt-en-es`` (~74 M parameters) on the EN→ES
dataset using the same vanilla PyTorch loop as ``scripts/train.py``.

Usage::

    python scripts/train_pretrained.py \\
        --data-config configs/data/config.yaml \\
        --train-config configs/train/text_pretrained.yaml

Comparison with Option A (``train.py``):

==================================  ========================================
Option A (train.py)                 Option B (this script)
==================================  ========================================
Trains from random weights          Fine-tunes pre-trained weights
Needs many epochs to converge       Reaches >20 BLEU in 1–2 epochs
~65–220 M parameters                ~74 M parameters (fixed)
Custom BPE tokeniser                MarianTokenizer (pre-built vocab)
Full architectural understanding    Quick pipeline sanity check
==================================  ========================================

Use Option B first to verify the pipeline works: if BLEU does not improve
after one fine-tuning epoch there is almost certainly a bug in the training
loop (loss computation, mask handling, etc.), not in the model itself.

Fine-tuning tips:

- Use a **lower learning rate** than training from scratch (5e-5 vs 1e-4).
  The pre-trained weights are already good; a large LR would overwrite them
  catastrophically ("catastrophic forgetting").
- Use **fewer epochs** (2–3 is typically enough). The model already understands
  translation; you are adapting it to your domain/style, not teaching from
  scratch.

**Important — tokeniser difference.** Option B uses ``MarianTokenizer`` (loaded
here), not the custom BPE tokeniser from ``src/text.py``. The pre-trained model
was trained with its own 60 k-vocabulary tokeniser and *cannot* be used with a
different vocabulary. Students must be aware of this when comparing A and B.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import MarianTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model_pretrained import PretrainedTranslationModel
from src.plotting import TrainingTracker
from src.utils import (
    compute_bleu,
    count_parameters,
    create_logger,
    load_checkpoint,
    load_yaml_config,
    save_checkpoint,
)

logger = create_logger(__name__)

MODEL_NAME = "Helsinki-NLP/opus-mt-en-es"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the fine-tuning script.

    Returns:
        argparse.Namespace: Parsed arguments with attributes
            ``data_config``, ``train_config``, ``resume``, ``device``,
            ``lr``, ``epochs``, and ``model_name``.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune Helsinki-NLP/opus-mt-en-es on EN→ES data (Option B)."
    )
    parser.add_argument("--data-config", type=str, default="configs/data/config.yaml")
    parser.add_argument("--train-config", type=str, default="configs/train/text_pretrained.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Peak learning rate (default: read from --train-config).",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of fine-tuning epochs (default: read from --train-config).",
    )
    parser.add_argument(
        "--model-name", type=str, default=MODEL_NAME,
        help="HuggingFace model identifier for the pre-trained model.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory for training curve plots and metrics JSON (default: output/).",
    )
    return parser.parse_args()


class MarianTranslationDataset(Dataset):
    """In-memory dataset that stores raw text strings for MarianTokenizer.

    Unlike ``TranslationDataset`` (which loads custom BPE token IDs from disk),
    this dataset stores raw text and tokenises it on-the-fly inside
    ``collate_fn``. This allows using ``MarianTokenizer`` directly without
    re-running the full preprocessing pipeline from ``src/text.py``.

    For very large datasets, consider pre-tokenising and saving to disk to
    avoid repeated tokenisation overhead.

    Args:
        src_texts (list[str]): English source strings.
        tgt_texts (list[str]): Spanish target strings.

    Raises:
        AssertionError: If ``src_texts`` and ``tgt_texts`` have different
            lengths.
    """

    def __init__(self, src_texts: list[str], tgt_texts: list[str]) -> None:
        assert len(src_texts) == len(tgt_texts), "Source and target must be the same length."
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts

    def __len__(self) -> int:
        """Return the number of sentence pairs.

        Returns:
            int: Dataset size.
        """
        return len(self.src_texts)

    def __getitem__(self, idx: int) -> dict:
        """Return a raw string pair.

        Args:
            idx (int): Example index.

        Returns:
            dict: Keys ``"src"`` (str) and ``"tgt"`` (str).
        """
        return {"src": self.src_texts[idx], "tgt": self.tgt_texts[idx]}


def make_marian_collate_fn(tokenizer: MarianTokenizer, max_seq_len: int):
    """Build a collate function that tokenises raw strings with MarianTokenizer.

    ``MarianTokenizer`` handles padding, truncation, and attention mask
    construction automatically, so no custom ``BucketSampler`` is needed.

    The ``as_target_tokenizer()`` context ensures the correct vocabulary is
    used for the target side. For ``opus-mt-en-es`` source and target share the
    same vocabulary, so this is a no-op, but it is still good practice and is
    required for MarianMT variants that use different vocabularies per side.

    Padding positions in ``labels`` are replaced with ``-100`` because
    HuggingFace models use ``-100`` as the conventional ``ignore_index`` for
    cross-entropy loss.

    Args:
        tokenizer (MarianTokenizer): Loaded tokeniser for the pre-trained model.
        max_seq_len (int): Maximum sequence length for truncation.

    Returns:
        Callable: A collate function suitable for use as
            ``DataLoader(collate_fn=...)``.
    """
    def collate(batch: list[dict]) -> dict:
        """Tokenise a batch of raw string pairs.

        Args:
            batch (list[dict]): List of ``{"src": str, "tgt": str}`` dicts
                returned by ``MarianTranslationDataset.__getitem__``.

        Returns:
            dict: Keys ``"input_ids"``, ``"attention_mask"``, ``"labels"``,
                and ``"tgt_ids_raw"`` (for BLEU decoding).
        """
        src_texts = [item["src"] for item in batch]
        tgt_texts = [item["tgt"] for item in batch]

        src_encoding = tokenizer(
            src_texts,
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )

        # as_target_tokenizer() was removed in transformers >= 4.35.
        # Pass text_target= directly; for opus-mt-en-es the vocabulary is
        # shared so this is equivalent.
        tgt_encoding = tokenizer(
            text_target=tgt_texts,
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )

        labels = tgt_encoding["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100

        return {
            "input_ids": src_encoding["input_ids"],
            "attention_mask": src_encoding["attention_mask"],
            "labels": labels,
            "tgt_ids_raw": tgt_encoding["input_ids"],
        }

    return collate


def load_raw_text_dataset(
    cfg_data: dict,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Load raw English/Spanish text from the Arrow dataset on disk.

    Reads from ``data/raw/`` (produced by ``src/text.py`` Phase 1) to obtain
    the original text strings. ``MarianTokenizer`` requires text, not custom
    BPE integer IDs.

    Args:
        cfg_data (dict): Data config dict with ``"raw_data_dir"``,
            ``"src_lang_col"``, and ``"tgt_lang_col"`` keys.

    Returns:
        tuple[list[str], list[str], list[str], list[str]]:
            ``(train_src, train_tgt, val_src, val_tgt)`` — lists of strings.
    """
    import datasets as hf_datasets

    raw_dir = cfg_data["raw_data_dir"]
    src_col = cfg_data["src_lang_col"]
    tgt_col = cfg_data["tgt_lang_col"]

    logger.info(f"Loading raw text from '{raw_dir}' …")
    dataset_dict = hf_datasets.load_from_disk(raw_dir)

    train_src = dataset_dict["train"][src_col]
    train_tgt = dataset_dict["train"][tgt_col]
    val_src = dataset_dict["val"][src_col]
    val_tgt = dataset_dict["val"][tgt_col]

    logger.info(
        f"Raw text loaded — train: {len(train_src):,}, val: {len(val_src):,} pairs."
    )
    return train_src, train_tgt, val_src, val_tgt


@torch.no_grad()
def validate(
    model: PretrainedTranslationModel,
    val_loader: DataLoader,
    device: torch.device,
    tokenizer: MarianTokenizer,
    epoch: int,
    compute_bleu_flag: bool,
    label_smoothing: float = 0.0,
) -> dict:
    """Run validation; optionally compute greedy BLEU on the first 2 000 examples.

    Args:
        model (PretrainedTranslationModel): The model to evaluate.
        val_loader (DataLoader): Validation DataLoader.
        device (torch.device): Target device.
        tokenizer (MarianTokenizer): Tokeniser for decoding generated IDs.
        epoch (int): Current epoch number.
        compute_bleu_flag (bool): Whether to run greedy BLEU decoding.
        label_smoothing (float, optional): Passed to
            ``PretrainedTranslationModel.forward()`` so val loss matches
            training loss. Defaults to ``0.0``.

    Returns:
        dict: Keys ``"val_loss"`` (float) and ``"bleu"`` (float or ``None``).
    """
    model.eval()

    total_loss = 0.0
    n_batches = 0
    hypotheses: list[str] = []
    references: list[list[str]] = []

    val_bar = tqdm(val_loader, desc=f"  Val epoch {epoch}", leave=False)

    for batch in val_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        loss, _ = model(input_ids, attention_mask, labels,
                        label_smoothing=label_smoothing)
        total_loss += loss.item()
        n_batches += 1

        if compute_bleu_flag and len(hypotheses) < 2000:
            generated = model.generate(
                input_ids, attention_mask,
                max_new_tokens=128,
                num_beams=1,
            )
            hyp_strs = tokenizer.batch_decode(generated, skip_special_tokens=True)
            ref_strs = tokenizer.batch_decode(batch["tgt_ids_raw"].tolist(), skip_special_tokens=True)
            for hyp, ref in zip(hyp_strs, ref_strs):
                hypotheses.append(hyp)
                references.append([ref])

    avg_val_loss = total_loss / max(n_batches, 1)
    bleu_score = None

    if compute_bleu_flag and hypotheses:
        bleu_score = compute_bleu(hypotheses, references)
        logger.info(f"  BLEU (greedy, first 2k val examples): {bleu_score:.2f}")

    model.train()
    return {"val_loss": avg_val_loss, "bleu": bleu_score}


def train_one_epoch(
    model: PretrainedTranslationModel,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    global_step: int,
    max_grad_norm: float,
    log_interval: int,
    tokenizer: MarianTokenizer,
    label_smoothing: float = 0.0,
    tracker: "TrainingTracker | None" = None,
) -> dict:
    """Run one fine-tuning epoch and return performance metrics.

    The loss is returned directly by ``PretrainedTranslationModel.forward()``.

    Args:
        model (PretrainedTranslationModel): Model in training mode.
        train_loader (DataLoader): Training DataLoader.
        optimizer (Optimizer): AdamW optimiser.
        scheduler: LR scheduler; stepped once per batch.
        device (torch.device): Target device.
        epoch (int): Current epoch number.
        global_step (int): Global step counter at the start of this epoch.
        max_grad_norm (float): Gradient clipping threshold.
        log_interval (int): Log a metrics line every this many steps.
        tokenizer (MarianTokenizer): Used to count real tokens for throughput.
        label_smoothing (float, optional): Passed to
            ``PretrainedTranslationModel.forward()`` for smoothed CE loss.
            Defaults to ``0.0``.
        tracker ("TrainingTracker | None", optional): Optional tracker for
            logging step metrics. Defaults to ``None``.

    Returns:
        dict: Keys ``"avg_loss"`` (float), ``"avg_tokens_per_sec"`` (float),
            ``"epoch_time"`` (float), and ``"global_step"`` (int).
    """
    model.train()

    total_loss = 0.0
    n_batches = 0
    running_tokens_per_sec = 0.0
    epoch_start = time.perf_counter()

    batch_bar = tqdm(train_loader, desc=f"Epoch {epoch:02d}", leave=False, position=1)

    for batch_idx, batch in enumerate(batch_bar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        batch_start = time.perf_counter()

        optimizer.zero_grad()

        loss, _logits = model(
            input_ids, attention_mask, labels,
            label_smoothing=label_smoothing,
        )

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_grad_norm
        ).item()

        optimizer.step()
        scheduler.step()

        batch_time = time.perf_counter() - batch_start

        n_real_src = attention_mask.sum().item()
        n_real_tgt = (labels != -100).sum().item()
        tokens_per_sec = (n_real_src + n_real_tgt) / max(batch_time, 1e-9)

        total_loss += loss.item()
        n_batches += 1
        running_tokens_per_sec += tokens_per_sec
        global_step += 1

        current_lr = scheduler.get_last_lr()[0]
        batch_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            tok_s=f"{tokens_per_sec:,.0f}",
            lr=f"{current_lr:.2e}",
            gnorm=f"{grad_norm:.2f}",
        )

        if global_step % log_interval == 0:
            elapsed = str(datetime.timedelta(seconds=int(time.perf_counter() - epoch_start)))
            logger.info(
                f"[Epoch {epoch:02d}] [Step {global_step:6d}] "
                f"loss={loss.item():.4f}  tok/s={tokens_per_sec:>8,.0f}  "
                f"lr={current_lr:.2e}  grad_norm={grad_norm:.3f}  elapsed={elapsed}"
            )
            if tracker is not None:
                tracker.log_step(global_step, loss.item(), current_lr)

    epoch_time = time.perf_counter() - epoch_start
    return {
        "avg_loss": total_loss / max(n_batches, 1),
        "avg_tokens_per_sec": running_tokens_per_sec / max(n_batches, 1),
        "epoch_time": epoch_time,
        "global_step": global_step,
    }


def main() -> None:
    """Orchestrate the full fine-tuning run.

    Loads configs and CLI overrides, builds DataLoaders (raw text, no
    BucketSampler), loads the pre-trained model, builds a linear warmup+decay
    scheduler, and runs the epoch loop. Saves checkpoints and handles emergency
    saves on ``KeyboardInterrupt``.

    The LR schedule uses linear warmup over 6 % of total steps, then linear
    decay to 0. The pre-trained model is already near a good optimum; a simple decay suffices.
    """
    args = parse_args()

    cfg_data = load_yaml_config(args.data_config)
    cfg_train = load_yaml_config(args.train_config)

    num_epochs = args.epochs if args.epochs is not None else cfg_train["training"]["num_epochs"]
    lr = args.lr if args.lr is not None else cfg_train["training"]["learning_rate"]
    warmup_steps_cfg = cfg_train["training"]["warmup_steps"]
    max_seq_len = cfg_data["max_seq_len"]
    max_grad_norm = cfg_train["training"]["max_grad_norm"]
    log_interval = cfg_train["logging"]["log_every_n_steps"]
    batch_size = cfg_train["training"]["batch_size"]
    ckpt_cfg = cfg_train["checkpointing"]

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    logger.info(f"Loading MarianTokenizer for '{args.model_name}' …")
    tokenizer = MarianTokenizer.from_pretrained(args.model_name)

    train_src, train_tgt, val_src, val_tgt = load_raw_text_dataset(cfg_data)

    collate = make_marian_collate_fn(tokenizer, max_seq_len)

    train_loader = DataLoader(
        MarianTranslationDataset(train_src, train_tgt),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=cfg_data["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        MarianTranslationDataset(val_src, val_tgt),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=cfg_data["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(f"Loading pre-trained model: {args.model_name} …")
    model = PretrainedTranslationModel(model_name=args.model_name)
    model = model.to(device)
    count_parameters(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )

    total_steps = len(train_loader) * num_epochs
    warmup_steps = warmup_steps_cfg

    def linear_schedule(step: int) -> float:
        """Linear warmup then linear decay to 0.

        Args:
            step (int): Current optimiser step.

        Returns:
            float: LR multiplier in ``[0, 1]``.
        """
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_schedule)

    start_epoch = 1
    global_step = 0

    run_name = f"pretrained_{os.path.splitext(os.path.basename(args.train_config))[0]}"
    tracker = TrainingTracker(output_dir=args.output_dir, run_name=run_name)

    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        meta = load_checkpoint(args.resume, model, optimizer, device=device)
        start_epoch = meta["epoch"] + 1
        global_step = meta["step"]
        logger.info(f"Resumed at epoch {meta['epoch']}, step {meta['step']}.")

    logger.info(f"Starting fine-tuning: {num_epochs} epochs, lr={lr}.")

    epoch_bar = tqdm(range(start_epoch, num_epochs + 1), desc="Fine-tuning", position=0)

    try:
        for epoch in epoch_bar:
            train_metrics = train_one_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                epoch=epoch,
                global_step=global_step,
                max_grad_norm=max_grad_norm,
                log_interval=log_interval,
                tokenizer=tokenizer,
                label_smoothing=cfg_train["training"]["label_smoothing"],
                tracker=tracker,
            )
            global_step = train_metrics["global_step"]

            do_bleu = (epoch % cfg_train["logging"]["compute_bleu_every_n_epochs"] == 0)
            val_metrics = validate(
                model=model,
                val_loader=val_loader,
                device=device,
                tokenizer=tokenizer,
                epoch=epoch,
                compute_bleu_flag=do_bleu,
                label_smoothing=cfg_train["training"]["label_smoothing"],
            )

            elapsed_str = str(
                datetime.timedelta(seconds=int(train_metrics["epoch_time"]))
            )
            bleu_str = (
                f"  BLEU={val_metrics['bleu']:.2f}" if val_metrics["bleu"] else ""
            )
            logger.info(
                f"[Epoch {epoch:02d}/{num_epochs}] COMPLETE | "
                f"train_loss={train_metrics['avg_loss']:.4f}  "
                f"val_loss={val_metrics['val_loss']:.4f}  "
                f"tok/s={train_metrics['avg_tokens_per_sec']:,.0f}"
                f"{bleu_str}  "
                f"epoch_time={elapsed_str}"
            )

            epoch_bar.set_postfix(
                train_loss=f"{train_metrics['avg_loss']:.4f}",
                val_loss=f"{val_metrics['val_loss']:.4f}",
            )

            tracker.log_epoch(
                epoch=epoch,
                train_loss=train_metrics["avg_loss"],
                val_loss=val_metrics["val_loss"],
                tokens_per_sec=train_metrics["avg_tokens_per_sec"],
                bleu=val_metrics["bleu"],
            )
            tracker.save_plots()

            if epoch % ckpt_cfg["save_every_n_epochs"] == 0:
                ckpt_path = os.path.join(
                    ckpt_cfg["checkpoint_dir"],
                    f"pretrained_checkpoint_ep{epoch:03d}_step{global_step:07d}.pt",
                )
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    step=global_step,
                    loss=val_metrics["val_loss"],
                    path=ckpt_path,
                    cfg=cfg_train,
                    keep_last_n=ckpt_cfg["keep_last_n"],
                )
                logger.info(f"Checkpoint saved: {ckpt_path}")

    except KeyboardInterrupt:
        logger.warning("Fine-tuning interrupted. Saving emergency checkpoint …")
        emergency_path = os.path.join(
            ckpt_cfg["checkpoint_dir"],
            f"pretrained_interrupted_ep{epoch}_step{global_step}.pt",
        )
        save_checkpoint(
            model=model, optimizer=optimizer, epoch=epoch,
            step=global_step, loss=float("inf"),
            path=emergency_path, cfg=cfg_train, keep_last_n=0,
        )
        logger.info(f"Emergency checkpoint saved: {emergency_path}")

    logger.info("Fine-tuning complete.")


if __name__ == "__main__":
    main()
