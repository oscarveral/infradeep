#!/usr/bin/env python
"""Data download pipeline for the EN→ES translation dataset.

Downloads ``Iker/OpenHermes-2.5-English-Spanish`` from HuggingFace Hub,
flattens the nested conversation format into parallel ``(en, es)`` text
pairs, and saves three Arrow splits to ``data/text/``.

Run once before benchmarking::

    python scripts/text/dataset.py

Use ``--inspect`` to print column names and sample rows without saving.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml
import datasets
from datasets import DatasetDict

# ------------------------------------------------------------------ #
#  Defaults (matching MarianMTConfig)
# ------------------------------------------------------------------ #

DEFAULTS = {
    "dataset_name": "Iker/OpenHermes-2.5-English-Spanish",
    "dataset_split": "train",
    "src_lang_col": "en",
    "tgt_lang_col": "es",
    "raw_data_dir": str(Path(__file__).resolve().parents[2] / "data" / "text"),
    "val_ratio": 0.05,
    "test_ratio": 0.05,
    "split_seed": 42,
}

# ------------------------------------------------------------------ #
#  Logger
# ------------------------------------------------------------------ #

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def download_dataset(cfg: dict) -> DatasetDict:
    """Download and split the dataset; save raw Arrow files to ``raw_data_dir``.

    Idempotent: if the data already exists on disk it is loaded directly.
    """
    raw_dir = cfg["raw_data_dir"]
    dataset_info_path = os.path.join(raw_dir, "dataset_dict.json")

    if os.path.exists(dataset_info_path):
        logger.info(f"Raw dataset already exists at '{raw_dir}'. Loading from disk.")
        return datasets.load_from_disk(raw_dir)

    logger.info(f"Downloading dataset '{cfg['dataset_name']}' …")
    dataset = datasets.load_dataset(cfg["dataset_name"], split=cfg["dataset_split"])

    logger.info(f"Dataset columns  : {dataset.column_names}")
    logger.info(f"Dataset features : {dataset.features}")
    logger.info(f"Dataset size     : {len(dataset):,} examples")

    dataset = _ensure_flat_columns(dataset, cfg)

    val_test_size = cfg["val_ratio"] + cfg["test_ratio"]
    split1 = dataset.train_test_split(test_size=val_test_size, seed=cfg["split_seed"])

    relative_test = cfg["test_ratio"] / val_test_size
    split2 = split1["test"].train_test_split(test_size=relative_test, seed=cfg["split_seed"])

    dataset_dict = DatasetDict({
        "train": split1["train"],
        "val": split2["train"],
        "test": split2["test"],
    })

    logger.info(
        f"Split sizes — train: {len(dataset_dict['train']):,}  "
        f"val: {len(dataset_dict['val']):,}  "
        f"test: {len(dataset_dict['test']):,}"
    )

    os.makedirs(raw_dir, exist_ok=True)
    dataset_dict.save_to_disk(raw_dir)
    logger.info(f"Raw dataset saved to '{raw_dir}'.")

    return dataset_dict


def _ensure_flat_columns(dataset: datasets.Dataset, cfg: dict) -> datasets.Dataset:
    """Normalise the raw dataset into flat ``(en, es)`` string columns.

    Extracts only human-authored turns from the nested conversation format.
    """
    src_col = cfg["src_lang_col"]
    tgt_col = cfg["tgt_lang_col"]

    if src_col in dataset.column_names and tgt_col in dataset.column_names:
        sample = dataset[0]
        if isinstance(sample[src_col], str) and isinstance(sample[tgt_col], str):
            return dataset

    if (
        "conversations_english" in dataset.column_names
        and "conversations_spanish" in dataset.column_names
    ):
        logger.info(
            "Detected conversations_english/conversations_spanish format. "
            "Extracting human-written turns only (from='human'). "
            f"Producing flat '{src_col}' / '{tgt_col}' columns."
        )

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
            desc="Extracting human turns from EN/ES conversations",
        )
        logger.info(f"Extracted {len(dataset):,} human turn pairs.")
        return dataset

    raise ValueError(
        f"Cannot find source column '{src_col}' or target column '{tgt_col}' "
        f"in the dataset, and no known nested format was detected. "
        f"Available columns: {dataset.column_names}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Data download for the EN→ES translation benchmark."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config file to override defaults.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Download the dataset, print column names and samples, then exit.",
    )
    args = parser.parse_args()

    cfg = dict(DEFAULTS)
    if args.config:
        with open(args.config, "r") as f:
            overrides = yaml.safe_load(f) or {}
        cfg.update(overrides)

    dataset = download_dataset(cfg)

    if args.inspect:
        print("\n" + "=" * 60)
        print("DATASET INSPECTION")
        print("=" * 60)
        print(f"Splits      : {list(dataset.keys())}")
        print(f"Columns     : {dataset['train'].column_names}")
        print(f"Train size  : {len(dataset['train']):,}")
        print()
        print("First 3 examples (train split)")
        for i in range(min(3, len(dataset["train"]))):
            row = dataset["train"][i]
            print(f"\n[{i}]")
            for col, val in row.items():
                print(f"  {col}: {str(val)[:200]}")
        print("=" * 60)
        return

    logger.info("Data pipeline complete. Raw text saved to data/raw/.")


if __name__ == "__main__":
    main()
