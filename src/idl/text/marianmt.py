import glob
import time
import statistics
from contextlib import nullcontext
from dataclasses import dataclass, field
import os

os.environ["HF_TOKEN"] = ""
os.environ["HF_HUB_VERBOSITY"] = "error"

import torch
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from accelerate import Accelerator, DistributedDataParallelKwargs
import datasets as hf_datasets

hf_datasets.disable_progress_bars()
hf_datasets.logging.set_verbosity_error()

import sacrebleu

from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, Dataset
from transformers import MarianMTModel, MarianTokenizer
import transformers

transformers.logging.disable_progress_bar()
transformers.logging.set_verbosity_error()


def _ensure_marian_positional_patch():
    """Apply a runtime monkeypatch to MarianSinusoidalPositionalEmbedding.create_weight
    to be robust against unexpected 1D `weight` tensors in some environments.
    This touches only project files and runs before any model is loaded.
    """
    try:
        from transformers.models.marian import modeling_marian
    except Exception:
        return

    # If patch already applied, skip
    if getattr(modeling_marian.MarianSinusoidalPositionalEmbedding.create_weight, "_patched", False):
        return

    def _create_weight(self):
        # Prefer explicit embedding attributes; fall back to weight.shape for compatibility.
        n_pos = getattr(self, "num_embeddings", None)
        dim = getattr(self, "embedding_dim", None)
        if n_pos is None or dim is None:
            w = getattr(self, "weight", None)
            if w is None:
                raise RuntimeError("MarianSinusoidalPositionalEmbedding has no weight or embedding attributes")
            if hasattr(w, "shape") and len(w.shape) == 2:
                n_pos, dim = w.shape
            elif hasattr(w, "shape") and len(w.shape) == 1:
                # fallback: infer from available attrs or assume embedding_dim equals length
                dim = dim or int(w.shape[0])
                n_pos = n_pos or 1
            else:
                raise RuntimeError("Unable to determine positional embedding shape")

        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out = torch.empty(n_pos, dim, dtype=getattr(self, "weight", torch.tensor(0)).dtype, requires_grad=False)
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        return out

    _create_weight._patched = True
    modeling_marian.MarianSinusoidalPositionalEmbedding.create_weight = _create_weight

from idl.accelerate import ProfileConfig
from idl.text.config import MarianMTConfig


# ------------------------------------------------------------------ #
#  Dataset & Collator
# ------------------------------------------------------------------ #

class TranslationDataset(Dataset):
    """In-memory dataset of raw text pairs for MarianTokenizer."""

    def __init__(self, src_texts: list[str], tgt_texts: list[str]) -> None:
        assert len(src_texts) == len(tgt_texts)
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts

    def __len__(self) -> int:
        return len(self.src_texts)

    def __getitem__(self, idx: int) -> dict:
        return {"src": self.src_texts[idx], "tgt": self.tgt_texts[idx]}


@dataclass
class TranslationCollator:
    """Tokenizes raw text pairs on-the-fly using MarianTokenizer."""

    model_name: str
    max_seq_len: int
    with_labels: bool
    _tokenizer: MarianTokenizer | None = field(default=None, init=False, repr=False)

    def _get_tokenizer(self) -> MarianTokenizer:
        if self._tokenizer is None:
            self._tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    def __call__(self, batch: list[dict]) -> dict:
        tokenizer = self._get_tokenizer()
        src_texts = [item["src"] for item in batch]

        src_enc = tokenizer(
            src_texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )

        if not self.with_labels:
            return {
                "input_ids": src_enc["input_ids"],
                "attention_mask": src_enc["attention_mask"],
            }

        tgt_texts = [item["tgt"] for item in batch]
        tgt_enc = tokenizer(
            text_target=tgt_texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )

        labels = tgt_enc["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100

        return {
            "input_ids": src_enc["input_ids"],
            "attention_mask": src_enc["attention_mask"],
            "labels": labels,
        }


# ------------------------------------------------------------------ #
#  Benchmark class
# ------------------------------------------------------------------ #

class MarianMT:
    model_config: MarianMTConfig = None
    profile_config: ProfileConfig = None
    accelerator: Accelerator = None
    model: MarianMTModel = None
    tokenizer: MarianTokenizer = None
    train_dataset: TranslationDataset = None
    train_dataloader: DataLoader = None
    val_dataset: TranslationDataset = None
    val_dataloader: DataLoader = None
    optimizer: Optimizer = None
    lr_scheduler = None

    def __init__(self, config: MarianMTConfig, profile: ProfileConfig):
        self.model_config = config
        self.profile_config = profile

    # ------------------------------------------------------------------ #
    #  Model loading
    # ------------------------------------------------------------------ #

    def load_model(self):
        if self.model_config.mode == "train":
            self.load_model_training()
        elif self.model_config.mode == "inference":
            self.load_model_inference()
        else:
            raise ValueError(f"Unknown mode: {self.model_config.mode}")

    def load_model_training(self):

        # Recreate the accelerator with DDP find_unused_parameters=True
        # because MarianMT has parameters that don't participate in the loss calculation.
        base_accelerator = self.profile_config.accelerator()
        kwargs_handlers = base_accelerator.state.kwargs_handlers if hasattr(base_accelerator.state, "kwargs_handlers") else []
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        
        # We need to extract the existing Keyword args or just rely on passing it.
        # However, a cleaner way is to just instantiate Accelerator directly with 
        # the merged kwargs if needed, or modify self.profile_config.accelerator() 
        # to accept extra kwargs. Since modifying accelerate.py might affect vit.py, 
        # we will do it here.
        
        handlers = []
        if self.profile_config.generate_profile_kwargs() is not None:
            handlers.append(self.profile_config.generate_profile_kwargs())
        handlers.append(ddp_kwargs)
        
        activities = self.profile_config.activities or []
        normalized_activities = {str(a).lower() for a in activities}
        force_cpu = "cpu" in normalized_activities and "cuda" not in normalized_activities
        
        self.accelerator = Accelerator(kwargs_handlers=handlers, cpu=force_cpu)
        _ensure_marian_positional_patch()
        self.model = MarianMTModel.from_pretrained(self.model_config.model_name)
        self.model.generation_config.max_length = None
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_config.model_name)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.model_config.learning_rate,
            weight_decay=0.01,
        )

        train_src, train_tgt, val_src, val_tgt = self._load_raw_text()
        self.train_dataset = TranslationDataset(train_src, train_tgt)
        self.val_dataset = TranslationDataset(val_src, val_tgt)

        collator = TranslationCollator(
            model_name=self.model_config.model_name,
            max_seq_len=self.model_config.max_seq_len,
            with_labels=True,
        )

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.model_config.per_device_batch_size,
            collate_fn=collator,
            shuffle=True,
            num_workers=self.model_config.num_workers,
            pin_memory=self.model_config.pin_memory,
            persistent_workers=self.model_config.persistent_workers,
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.model_config.per_device_batch_size,
            collate_fn=collator,
            num_workers=self.model_config.num_workers,
            pin_memory=self.model_config.pin_memory,
            persistent_workers=self.model_config.persistent_workers,
        )

        total_steps = max(1, len(self.train_dataloader) * self.model_config.num_epochs)
        warmup_steps = self.model_config.warmup_steps

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 1.0 - progress)

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.lr_scheduler = (
            self.accelerator.prepare(
                self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.lr_scheduler
            )
        )

    def load_model_inference(self):
        self.accelerator = self.profile_config.accelerator()

        _ensure_marian_positional_patch()
        self.model = MarianMTModel.from_pretrained(self.model_config.model_name)
        self.model.generation_config.max_length = None
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_config.model_name)
        self.model.eval()

        _, _, val_src, val_tgt = self._load_raw_text()
        self.val_dataset = TranslationDataset(val_src, val_tgt)

        collator = TranslationCollator(
            model_name=self.model_config.model_name,
            max_seq_len=self.model_config.max_seq_len,
            with_labels=False,
        )

        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.model_config.per_device_batch_size,
            collate_fn=collator,
            num_workers=self.model_config.num_workers,
            pin_memory=self.model_config.pin_memory,
            persistent_workers=self.model_config.persistent_workers,
        )

        self.model, self.val_dataloader = self.accelerator.prepare(self.model, self.val_dataloader)

    # ------------------------------------------------------------------ #
    #  Training
    # ------------------------------------------------------------------ #

    def run_train(self):
        if self.train_dataloader is None or self.val_dataloader is None:
            self.load_model_training()

        self.model.train()
        self._cuda_sync()
        self._reset_cuda_peak_memory()

        # ---- warmup phase (excluded from timing) ---- #
        warmup = self.model_config.warmup_batches
        if warmup > 0:
            warmup_iter = iter(self.train_dataloader)
            for _ in range(warmup):
                try:
                    batch = next(warmup_iter)
                except StopIteration:
                    break
                self.optimizer.zero_grad()
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                self.accelerator.backward(outputs.loss)
                self.optimizer.step()
                self.lr_scheduler.step()
            self._cuda_sync()

        # ---- timed phase ---- #
        start_time = time.perf_counter()
        step_metrics: list[dict] = []
        epoch_metrics: list[dict] = []
        timed_out = False

        memory_rss_samples_mb: list[float] = []
        memory_cuda_samples_mb: list[float] = []
        total_train_tokens = 0

        label_smoothing = self.model_config.label_smoothing
        criterion = None
        if label_smoothing > 0.0:
            criterion = torch.nn.CrossEntropyLoss(
                ignore_index=-100,
                label_smoothing=label_smoothing,
            )

        with self._profile_context() as prof:
            for epoch in range(self.model_config.num_epochs):
                total_train_loss = 0.0
                step_count = 0

                for step, batch in enumerate(self.train_dataloader):
                    step_start = time.perf_counter()

                    self.optimizer.zero_grad()
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )

                    if criterion is not None:
                        loss = criterion(
                            outputs.logits.reshape(-1, outputs.logits.size(-1)),
                            batch["labels"].reshape(-1),
                        )
                    else:
                        loss = outputs.loss

                    self.accelerator.backward(loss)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.model_config.max_grad_norm,
                    )
                    self.optimizer.step()
                    self.lr_scheduler.step()

                    self._cuda_sync()
                    batch_latency = time.perf_counter() - step_start

                    # Count real tokens (non-padding).
                    n_src_tokens = int(batch["attention_mask"].sum().item())
                    n_tgt_tokens = int((batch["labels"] != -100).sum().item())
                    batch_tokens = n_src_tokens + n_tgt_tokens
                    total_train_tokens += batch_tokens

                    step_elapsed = max(1e-9, time.perf_counter() - start_time)
                    tokens_per_second = total_train_tokens / step_elapsed

                    rss_mb = self._current_rss_mb()
                    if rss_mb is not None:
                        memory_rss_samples_mb.append(rss_mb)

                    cuda_mb = self._current_cuda_allocated_mb()
                    if cuda_mb is not None:
                        memory_cuda_samples_mb.append(cuda_mb)

                    current_lr = self.optimizer.param_groups[0]["lr"]
                    step_metrics.append(
                        {
                            "epoch": epoch + 1,
                            "step": step,
                            "train_loss": float(loss.item()),
                            "learning_rate": float(current_lr),
                            "batch_tokens": batch_tokens,
                            "batch_latency_seconds": round(batch_latency, 6),
                            "tokens_per_second": float(tokens_per_second),
                            "memory_rss_mb": float(rss_mb) if rss_mb is not None else None,
                            "memory_cuda_allocated_mb": float(cuda_mb) if cuda_mb is not None else None,
                        }
                    )

                    total_train_loss += float(loss.item())
                    step_count += 1

                    if self.profile_config.profile and prof is not None and hasattr(prof, "step"):
                        prof.step()

                    if step % 100 == 0:
                        print(f"Step {step}: train_loss={float(loss.item()):.4f}, learning_rate={current_lr:.2e}, batch_tokens={batch_tokens}, batch_latency_seconds={batch_latency:.4f}, tokens_per_second={tokens_per_second:.0f}, memory_rss_mb={rss_mb:.1f}, memory_cuda_allocated_mb={cuda_mb:.1f}")

                    if (time.perf_counter() - start_time) > self.model_config.time_limit_seconds:
                        timed_out = True
                        break

                    # If profiling. Just end after 300 steps.
                    if (self.profile_config.profile and step >= 100):
                        timed_out = True
                        break

                if timed_out:
                    break

                # ---- validation at end of epoch ---- #
                self.model.eval()
                total_val_loss = 0.0
                val_batches = 0
                bleu_hypotheses: list[str] = []
                bleu_references: list[list[str]] = []

                for batch in self.val_dataloader:
                    with torch.no_grad():
                        outputs = self.model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"],
                        )
                    total_val_loss += float(outputs.loss.item())
                    val_batches += 1

                    # BLEU: greedy decode on the first N val examples.
                    if len(bleu_hypotheses) < self.model_config.bleu_eval_samples:
                        unwrapped = self.accelerator.unwrap_model(self.model)
                        with torch.no_grad():
                            generated = unwrapped.generate(
                                input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                max_new_tokens=128,
                                num_beams=1,
                            )
                        hyps = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                        ref_ids = batch["labels"].clone()
                        ref_ids[ref_ids == -100] = self.tokenizer.pad_token_id
                        refs = self.tokenizer.batch_decode(ref_ids, skip_special_tokens=True)
                        for h, r in zip(hyps, refs):
                            bleu_hypotheses.append(h)
                            bleu_references.append([r])

                    if self.profile_config.profile and prof is not None and hasattr(prof, "step"):
                        prof.step()

                avg_train_loss = total_train_loss / max(1, step_count)
                avg_val_loss = total_val_loss / max(1, val_batches)

                bleu_score = None
                if bleu_hypotheses:
                    bleu_result = sacrebleu.corpus_bleu(bleu_hypotheses, bleu_references)
                    bleu_score = float(bleu_result.score)

                epoch_metrics.append(
                    {
                        "epoch": epoch + 1,
                        "train_loss": float(avg_train_loss),
                        "val_loss": float(avg_val_loss),
                        "bleu": bleu_score,
                    }
                )

                # Save checkpoint after each epoch.
                self._save_checkpoint(epoch + 1, step_count, avg_val_loss)

                self.model.train()
                if timed_out:
                    break

        self.accelerator.wait_for_everyone()
        self._cuda_sync()
        elapsed = time.perf_counter() - start_time

        global_train_tokens = int(
            self.accelerator.reduce(
                torch.tensor(total_train_tokens, device=self.accelerator.device),
                reduction="sum",
            ).item()
        )

        throughput = global_train_tokens / max(1e-9, elapsed)
        profile_stats = self._build_profile_stats(prof)
        traces = self.profile_config.consume_trace_payloads()

        return {
            "stats": {
                "mode": "train",
                "timed_out": timed_out,
                "elapsed_seconds": round(elapsed, 4),
                "warmup_batches": self.model_config.warmup_batches,
                "total_tokens_processed": int(global_train_tokens),
                "throughput_tokens_per_second": round(throughput, 4),
                "step_metrics": step_metrics,
                "epoch_metrics": epoch_metrics,
                "model_size": self._build_model_size_stats(),
                "memory": self._build_memory_stats(memory_rss_samples_mb, memory_cuda_samples_mb),
                "profile": profile_stats,
                "config": self._build_config_snapshot(),
            },
            "traces": {
                "chrome": traces,
            },
        }

    # ------------------------------------------------------------------ #
    #  Inference
    # ------------------------------------------------------------------ #

    def run_inference(self):
        if self.val_dataloader is None:
            self.load_model_inference()
        if self.model is not None:
            self.model.eval()

        self._cuda_sync()
        self._reset_cuda_peak_memory()

        # ---- warmup phase (excluded from timing) ---- #
        warmup = self.model_config.warmup_batches
        if warmup > 0:
            warmup_iter = iter(self.val_dataloader)
            for _ in range(warmup):
                try:
                    batch = next(warmup_iter)
                except StopIteration:
                    break
                with torch.no_grad():
                    _ = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
            self._cuda_sync()

        # ---- timed phase ---- #
        start_time = time.perf_counter()
        total_pairs = len(self.val_dataset) if self.val_dataset is not None else 0
        timed_out = False
        processed_tokens = 0
        memory_rss_samples_mb: list[float] = []
        memory_cuda_samples_mb: list[float] = []
        step_metrics: list[dict] = []

        with self._profile_context() as prof:
            for batch in self.val_dataloader:
                if (time.perf_counter() - start_time) > self.model_config.time_limit_seconds:
                    timed_out = True
                    break

                step_start = time.perf_counter()
                with torch.no_grad():
                    _ = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )

                self._cuda_sync()
                batch_latency = time.perf_counter() - step_start

                batch_tokens = int(batch["attention_mask"].sum().item())
                processed_tokens += batch_tokens

                step_elapsed = max(1e-9, time.perf_counter() - start_time)
                tokens_per_second = processed_tokens / step_elapsed

                rss_mb = self._current_rss_mb()
                if rss_mb is not None:
                    memory_rss_samples_mb.append(rss_mb)

                cuda_mb = self._current_cuda_allocated_mb()
                if cuda_mb is not None:
                    memory_cuda_samples_mb.append(cuda_mb)

                step_metrics.append(
                    {
                        "step": len(step_metrics),
                        "batch_tokens": batch_tokens,
                        "batch_latency_seconds": round(batch_latency, 6),
                        "processed_tokens": int(processed_tokens),
                        "tokens_per_second": float(tokens_per_second),
                        "memory_rss_mb": float(rss_mb) if rss_mb is not None else None,
                        "memory_cuda_allocated_mb": float(cuda_mb) if cuda_mb is not None else None,
                    }
                )

                if self.profile_config.profile and prof is not None and hasattr(prof, "step"):
                    prof.step()

        self.accelerator.wait_for_everyone()
        self._cuda_sync()

        elapsed = time.perf_counter() - start_time
        global_processed_tokens = int(
            self.accelerator.reduce(
                torch.tensor(processed_tokens, device=self.accelerator.device),
                reduction="sum",
            ).item()
        )
        throughput = (global_processed_tokens / elapsed) if elapsed > 0 else 0.0
        profile_stats = self._build_profile_stats(prof)
        traces = self.profile_config.consume_trace_payloads()

        return {
            "stats": {
                "mode": "inference",
                "timed_out": timed_out,
                "elapsed_seconds": round(elapsed, 4),
                "warmup_batches": self.model_config.warmup_batches,
                "total_pairs": int(total_pairs),
                "total_tokens_processed": int(global_processed_tokens),
                "throughput_tokens_per_second": round(throughput, 4),
                "step_metrics": step_metrics,
                "model_size": self._build_model_size_stats(),
                "memory": self._build_memory_stats(memory_rss_samples_mb, memory_cuda_samples_mb),
                "profile": profile_stats,
                "config": self._build_config_snapshot(),
            },
            "traces": {
                "chrome": traces,
            },
        }

    # ------------------------------------------------------------------ #
    #  Dispatch
    # ------------------------------------------------------------------ #

    def run(self):
        if self.model_config.mode == "train":
            return self.run_train()
        if self.model_config.mode == "inference":
            return self.run_inference()
        raise ValueError(f"Unknown mode: {self.model_config.mode}")

    # ------------------------------------------------------------------ #
    #  Data loading
    # ------------------------------------------------------------------ #

    def _load_raw_text(self) -> tuple[list[str], list[str], list[str], list[str]]:
        """Load raw text pairs from the Arrow dataset on disk.

        If the raw data does not exist, downloads and prepares it first.

        Returns:
            (train_src, train_tgt, val_src, val_tgt)
        """
        raw_dir = str(self.model_config.raw_data_dir)
        dataset_info = os.path.join(raw_dir, "dataset_dict.json")

        if not os.path.exists(dataset_info):
            self._download_dataset()

        dataset_dict = hf_datasets.load_from_disk(raw_dir)

        src_col = self.model_config.src_lang_col
        tgt_col = self.model_config.tgt_lang_col

        train_src = dataset_dict["train"][src_col]
        train_tgt = dataset_dict["train"][tgt_col]
        val_src = dataset_dict["val"][src_col]
        val_tgt = dataset_dict["val"][tgt_col]

        return train_src, train_tgt, val_src, val_tgt

    def _download_dataset(self):
        """Download and split the dataset, saving Arrow files to raw_data_dir."""
        raw_dir = str(self.model_config.raw_data_dir)
        dataset = hf_datasets.load_dataset(
            self.model_config.dataset_name,
            split="train",
        )

        # Flatten nested conversation format into (en, es) pairs.
        src_col = self.model_config.src_lang_col
        tgt_col = self.model_config.tgt_lang_col

        if "conversations_english" in dataset.column_names and "conversations_spanish" in dataset.column_names:
            def extract_human_turns(examples: dict) -> dict:
                src_texts: list[str] = []
                tgt_texts: list[str] = []
                for en_convs, es_convs in zip(
                    examples["conversations_english"],
                    examples["conversations_spanish"],
                ):
                    for en_turn, es_turn in zip(en_convs, es_convs):
                        if en_turn.get("from", "").lower() == "human":
                            en_text = en_turn.get("value", "").strip()
                            es_text = es_turn.get("value", "").strip()
                            if en_text and es_text:
                                src_texts.append(en_text)
                                tgt_texts.append(es_text)
                return {src_col: src_texts, tgt_col: tgt_texts}

            dataset = dataset.map(
                extract_human_turns,
                batched=True,
                batch_size=1000,
                remove_columns=dataset.column_names,
            )

        # Split into train / val / test.
        split1 = dataset.train_test_split(test_size=0.10, seed=42)
        split2 = split1["test"].train_test_split(test_size=0.50, seed=42)

        dataset_dict = hf_datasets.DatasetDict({
            "train": split1["train"],
            "val": split2["train"],
            "test": split2["test"],
        })

        os.makedirs(raw_dir, exist_ok=True)
        dataset_dict.save_to_disk(raw_dir)

    # ------------------------------------------------------------------ #
    #  Checkpointing
    # ------------------------------------------------------------------ #

    def _save_checkpoint(self, epoch: int, step: int, val_loss: float, keep_last_n: int = 3):
        """Save model checkpoint after an epoch. Only on main process."""
        if not self.accelerator.is_main_process:
            return

        ckpt_dir = self.model_config.checkpoint_dir
        os.makedirs(ckpt_dir, exist_ok=True)

        ckpt_path = os.path.join(
            ckpt_dir,
            f"marianmt_ep{epoch:03d}_step{step:07d}.pt",
        )

        unwrapped = self.accelerator.unwrap_model(self.model)
        state = {
            "epoch": epoch,
            "step": step,
            "val_loss": val_loss,
            "model_state_dict": unwrapped.state_dict(),
            "config": str(self.model_config),
        }
        torch.save(state, ckpt_path)

        # Rotate: keep only the last N checkpoints.
        if keep_last_n > 0:
            all_ckpts = sorted(
                glob.glob(os.path.join(ckpt_dir, "marianmt_ep*.pt")),
                key=os.path.getmtime,
            )
            for old in all_ckpts[:-keep_last_n]:
                os.remove(old)

    # ------------------------------------------------------------------ #
    #  Memory helpers
    # ------------------------------------------------------------------ #

    def _current_rss_mb(self) -> float | None:
        try:
            with open("/proc/self/status", "r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        kb = float(line.split()[1])
                        return kb / 1024.0
        except Exception:
            return None
        return None

    def _current_cuda_allocated_mb(self) -> float | None:
        if self.accelerator is not None and self.accelerator.device.type == "cuda":
            return torch.cuda.memory_allocated(self.accelerator.device) / (1024.0 * 1024.0)
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024.0 * 1024.0)
        return None

    def _cuda_peak_memory_mb(self) -> float | None:
        if self.accelerator is not None and self.accelerator.device.type == "cuda":
            return torch.cuda.max_memory_allocated(self.accelerator.device) / (1024.0 * 1024.0)
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        return None

    def _reset_cuda_peak_memory(self):
        if self.accelerator is not None and self.accelerator.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.accelerator.device)
        elif torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def _cuda_sync(self):
        if self.accelerator is not None and self.accelerator.device.type == "cuda":
            torch.cuda.synchronize(self.accelerator.device)

    # ------------------------------------------------------------------ #
    #  Statistics helpers
    # ------------------------------------------------------------------ #

    def _summarize_samples(self, samples: list[float]) -> dict | None:
        if not samples:
            return None
        result = {
            "min": float(min(samples)),
            "mean": float(statistics.fmean(samples)),
            "median": float(statistics.median(samples)),
            "max": float(max(samples)),
        }
        if len(samples) >= 2:
            result["stddev"] = float(statistics.stdev(samples))
        return result

    def _build_memory_stats(self, rss_samples: list[float], cuda_samples: list[float]) -> dict:
        stats = {
            "rss_mb": self._summarize_samples(rss_samples),
            "cuda_allocated_mb": self._summarize_samples(cuda_samples),
        }
        peak = self._cuda_peak_memory_mb()
        if peak is not None:
            stats["cuda_peak_mb"] = round(peak, 4)
        return stats

    def _build_model_size_stats(self) -> dict:
        """Compute model parameter counts and weight memory estimates."""
        if self.model is None:
            return {}
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable = total - trainable
        fp32_mb = total * 4 / (1024 ** 2)
        fp16_mb = total * 2 / (1024 ** 2)
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "non_trainable_parameters": non_trainable,
            "weight_memory_fp32_mb": round(fp32_mb, 2),
            "weight_memory_fp16_mb": round(fp16_mb, 2),
            "adam_state_fp32_mb": round(fp32_mb * 2, 2),
        }

    def _build_config_snapshot(self) -> dict:
        return {
            "model": str(self.model_config),
            "profile": str(self.profile_config),
        }

    # ------------------------------------------------------------------ #
    #  Profiling helpers
    # ------------------------------------------------------------------ #

    def _build_profile_stats(self, profiler):
        if not self.profile_config.profile:
            return {"enabled": False}

        stats = {"enabled": True}
        if profiler is None:
            return stats

        key_averages = profiler.key_averages()
        stats["key_averages_items"] = [str(item) for item in key_averages]
        stats["total_average"] = str(key_averages.total_average())

        return stats

    def _profile_context(self):
        if self.profile_config.profile:
            return self.accelerator.profile()
        return nullcontext()
