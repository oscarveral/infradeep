import time
import statistics
from contextlib import nullcontext
from dataclasses import dataclass, field
import os

os.environ["HF_TOKEN"] = ""
os.environ["HF_HUB_VERBOSITY"] = "error"

import torch
import warnings

warnings.filterwarnings("ignore")

from accelerate import Accelerator
from datasets import Dataset, load_dataset
import datasets

datasets.disable_progress_bars()
datasets.logging.set_verbosity_error()

from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor, get_scheduler
import transformers

transformers.logging.disable_progress_bar()
transformers.logging.set_verbosity_error()

from idl.accelerate import ProfileConfig
from idl.image.config import ViTConfig


@dataclass
class ViTCollator:
    model_name: str
    with_labels: bool
    _processor: ViTImageProcessor | None = field(default=None, init=False, repr=False)

    def __call__(self, items):
        if self._processor is None:
            self._processor = ViTImageProcessor.from_pretrained(self.model_name)

        images = [item["image"].convert("RGB") for item in items]
        inputs = self._processor(images=images, return_tensors="pt")
        if not self.with_labels:
            return inputs["pixel_values"]

        labels = torch.tensor([item["label"] for item in items])
        return {"pixel_values": inputs["pixel_values"], "labels": labels}


class ViT:
    model_config: ViTConfig = None
    profile_config: ProfileConfig = None
    accelerator: Accelerator = None
    model: ViTForImageClassification = None
    processor: ViTImageProcessor = None
    val_dataset: Dataset = None
    val_dataloader: DataLoader = None
    train_dataset: Dataset = None
    train_dataloader: DataLoader = None
    optimizer: Optimizer = None
    lr_scheduler = None

    def __init__(self, config: ViTConfig, profile: ProfileConfig):
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
        self.accelerator = self.profile_config.accelerator()

        self.model = ViTForImageClassification.from_pretrained(
            self.model_config.model_name,
            num_labels=self.model_config.num_labels,
            ignore_mismatched_sizes=True,
        )
        self.processor = ViTImageProcessor.from_pretrained(self.model_config.model_name)
        self.optimizer = AdamW(self.model.parameters(), lr=self.model_config.learning_rate)

        self.train_dataset = self._load_image_split(self.model_config.train_split)
        self.val_dataset = self._load_image_split(self.model_config.val_split)
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.model_config.per_device_batch_size,
            collate_fn=ViTCollator(self.model_config.model_name, with_labels=True),
            shuffle=True,
            num_workers=self.model_config.num_workers,
            pin_memory=self.model_config.pin_memory,
            persistent_workers=self.model_config.persistent_workers,
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.model_config.per_device_batch_size,
            collate_fn=ViTCollator(self.model_config.model_name, with_labels=True),
            num_workers=self.model_config.num_workers,
            pin_memory=self.model_config.pin_memory,
            persistent_workers=self.model_config.persistent_workers,
        )

        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=max(1, len(self.train_dataloader) * self.model_config.num_epochs),
        )

        self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.lr_scheduler = (
            self.accelerator.prepare(
                self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.lr_scheduler
            )
        )

    def load_model_inference(self):
        self.accelerator = self.profile_config.accelerator()

        self.model = ViTForImageClassification.from_pretrained(self.model_config.model_name)
        self.processor = ViTImageProcessor.from_pretrained(self.model_config.model_name)
        self.model.eval()

        self.val_dataset = self._load_image_split(self.model_config.val_split)
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.model_config.per_device_batch_size,
            collate_fn=ViTCollator(self.model_config.model_name, with_labels=False),
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
                outputs = self.model(pixel_values=batch["pixel_values"], labels=batch["labels"])
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
        total_train_images = 0

        with self._profile_context() as prof:
            for epoch in range(self.model_config.num_epochs):
                total_train_loss = 0.0
                step_count = 0

                for step, batch in enumerate(self.train_dataloader):
                    step_start = time.perf_counter()

                    self.optimizer.zero_grad()
                    outputs = self.model(pixel_values=batch["pixel_values"], labels=batch["labels"])
                    loss = outputs.loss

                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.lr_scheduler.step()

                    self._cuda_sync()
                    batch_latency = time.perf_counter() - step_start

                    batch_size = int(batch["labels"].size(0))
                    total_train_images += batch_size

                    step_elapsed = max(1e-9, time.perf_counter() - start_time)
                    images_per_second = total_train_images / step_elapsed

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
                            "batch_size": batch_size,
                            "batch_latency_seconds": round(batch_latency, 6),
                            "images_per_second": float(images_per_second),
                            "memory_rss_mb": float(rss_mb) if rss_mb is not None else None,
                            "memory_cuda_allocated_mb": float(cuda_mb) if cuda_mb is not None else None,
                        }
                    )

                    total_train_loss += float(loss.item())
                    step_count += 1

                    if self.profile_config.profile and prof is not None and hasattr(prof, "step"):
                        prof.step()

                    if (time.perf_counter() - start_time) > self.model_config.time_limit_seconds:
                        timed_out = True
                        break

                # ---- validation at end of epoch ---- #
                self.model.eval()
                total_val_loss = 0.0
                correct_predictions = 0
                total_predictions = 0

                for batch in self.val_dataloader:
                    with torch.no_grad():
                        outputs = self.model(
                            pixel_values=batch["pixel_values"],
                            labels=batch["labels"],
                        )

                    total_val_loss += float(outputs.loss.item())
                    predictions = outputs.logits.argmax(dim=-1)
                    correct_predictions += (predictions == batch["labels"]).sum().item()
                    total_predictions += batch["labels"].size(0)

                    if self.profile_config.profile and prof is not None and hasattr(prof, "step"):
                        prof.step()

                avg_train_loss = total_train_loss / max(1, step_count)
                avg_val_loss = total_val_loss / max(1, len(self.val_dataloader))
                accuracy = correct_predictions / max(1, total_predictions)

                epoch_metrics.append(
                    {
                        "epoch": epoch + 1,
                        "train_loss": float(avg_train_loss),
                        "val_loss": float(avg_val_loss),
                        "accuracy": float(accuracy),
                    }
                )

                self.model.train()
                if timed_out:
                    break

        self.accelerator.wait_for_everyone()
        self._cuda_sync()
        elapsed = time.perf_counter() - start_time

        global_train_images = int(
            self.accelerator.reduce(
                torch.tensor(total_train_images, device=self.accelerator.device),
                reduction="sum",
            ).item()
        )

        throughput = global_train_images / max(1e-9, elapsed)
        profile_stats = self._build_profile_stats(prof)
        traces = self.profile_config.consume_trace_payloads()

        return {
            "stats": {
                "mode": "train",
                "timed_out": timed_out,
                "elapsed_seconds": round(elapsed, 4),
                "warmup_batches": self.model_config.warmup_batches,
                "total_images_processed": int(global_train_images),
                "throughput_images_per_second": round(throughput, 4),
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
                    _ = self.model(pixel_values=batch)
            self._cuda_sync()

        # ---- timed phase ---- #
        start_time = time.perf_counter()
        total_images = len(self.val_dataset) if self.val_dataset is not None else 0
        timed_out = False
        processed_images = 0
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
                    _ = self.model(pixel_values=batch)

                self._cuda_sync()
                batch_latency = time.perf_counter() - step_start

                batch_size = int(batch.size(0))
                processed_images += batch_size

                step_elapsed = max(1e-9, time.perf_counter() - start_time)
                images_per_second = processed_images / step_elapsed

                rss_mb = self._current_rss_mb()
                if rss_mb is not None:
                    memory_rss_samples_mb.append(rss_mb)

                cuda_mb = self._current_cuda_allocated_mb()
                if cuda_mb is not None:
                    memory_cuda_samples_mb.append(cuda_mb)

                step_metrics.append(
                    {
                        "step": len(step_metrics),
                        "batch_size": batch_size,
                        "batch_latency_seconds": round(batch_latency, 6),
                        "processed_images": int(processed_images),
                        "images_per_second": float(images_per_second),
                        "memory_rss_mb": float(rss_mb) if rss_mb is not None else None,
                        "memory_cuda_allocated_mb": float(cuda_mb) if cuda_mb is not None else None,
                    }
                )

                if self.profile_config.profile and prof is not None and hasattr(prof, "step"):
                    prof.step()

        self.accelerator.wait_for_everyone()
        self._cuda_sync()

        elapsed = time.perf_counter() - start_time
        global_processed_images = int(
            self.accelerator.reduce(
                torch.tensor(processed_images, device=self.accelerator.device),
                reduction="sum",
            ).item()
        )
        throughput = (global_processed_images / elapsed) if elapsed > 0 else 0.0
        profile_stats = self._build_profile_stats(prof)
        traces = self.profile_config.consume_trace_payloads()

        return {
            "stats": {
                "mode": "inference",
                "timed_out": timed_out,
                "elapsed_seconds": round(elapsed, 4),
                "warmup_batches": self.model_config.warmup_batches,
                "total_images": int(total_images),
                "total_images_processed": int(global_processed_images),
                "throughput_images_per_second": round(throughput, 4),
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
    #  Memory helpers
    # ------------------------------------------------------------------ #

    def _current_rss_mb(self) -> float | None:
        """Linux-only: read current resident memory from /proc."""
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
        """Synchronize CUDA device if available, ensuring accurate timing."""
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

    def _build_memory_stats(
        self,
        rss_samples: list[float],
        cuda_samples: list[float],
    ) -> dict:
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
        """Embed resolved configs in output for reproducibility."""
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

    # ------------------------------------------------------------------ #
    #  Dataset helpers
    # ------------------------------------------------------------------ #

    def _load_image_split(self, split_name: str) -> Dataset:
        return load_dataset("imagefolder", data_dir=str(self.model_config.data_dir), split=split_name)
