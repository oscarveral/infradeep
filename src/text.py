"""Data download pipeline.

Downloads ``Iker/OpenHermes-2.5-English-Spanish`` from HuggingFace Hub,
flattens the nested conversation format into parallel ``(en, es)`` text
pairs, and saves three Arrow splits to ``data/raw/``.

Run once before training::

    python src/text.py --config configs/data/config.yaml

Use ``--inspect`` to print column names and sample rows without saving.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import datasets
from datasets import DatasetDict

from src.utils import create_logger, load_yaml_config

logger = create_logger(__name__)


def download_dataset(cfg: dict) -> DatasetDict:
    """Download and split the dataset; save raw Arrow files to ``data/raw/``.

    The function is idempotent: if ``data/raw/`` already contains a saved
    dataset it is loaded from disk and returned without re-downloading.

    The raw dataset is saved in Arrow columnar format, which supports
    memory-mapped access. DataLoader workers later read rows on demand via the
    OS page cache: only requested bytes are loaded into RAM, which avoids
    multiplying the dataset size by ``num_workers``.

    The published dataset has a single ``"train"`` split; this function carves
    out validation and test sets using the ratios in the config.

    Args:
        cfg (dict): Parsed data config from ``configs/data/config.yaml``.

    Returns:
        DatasetDict: Dataset with splits ``"train"``, ``"val"``, and ``"test"``.
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

    The ``Iker/OpenHermes-2.5-English-Spanish`` dataset stores each row as two
    parallel conversation lists::

        conversations_english: [{"from": "human"/"gpt", "value": "..."}, ...]
        conversations_spanish: [{"from": "human"/"gpt", "value": "..."}, ...]

    Turn ``i`` in ``conversations_english`` is the direct translation of turn
    ``i`` in ``conversations_spanish``. The lists are fully aligned by index.

    Only human-authored turns (``from='human'``) are extracted. GPT-generated
    responses are often long multi-paragraph answers that exceed
    ``max_seq_len=256`` and get heavily truncated, adding noise. Human turns
    are natural, concise sentences that make better MT training pairs.

    This function performs a flat-map: with ``batched=True`` and
    ``remove_columns``, ``datasets.map()`` supports returning more rows than it
    receives, emitting one ``(en, es)`` pair per human turn.

    Args:
        dataset (datasets.Dataset): Single-split HuggingFace dataset.
        cfg (dict): Data config dict with "src_lang_col" and
            "tgt_lang_col" keys.

    Returns:
        datasets.Dataset: Dataset with flat string columns named
            cfg["src_lang_col"] and cfg["tgt_lang_col"].

    Raises:
        ValueError: If neither the expected flat columns nor the
            "conversations_english"/"conversations_spanish" format are found.
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
            """Flat-map a batch of conversations to individual human-turn pairs.

            Args:
                examples (dict): Batched HuggingFace map format where each
                    value is a list; ``examples["conversations_english"]`` is a
                    list of conversations, each of which is a list of
                    ``{"from": str, "value": str}`` dicts.

            Returns:
                dict: Dict with ``src_col`` and ``tgt_col`` keys, each holding
                    a flat list of extracted strings.
            """
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
        f"Available columns: {dataset.column_names}. "
        "Update src_lang_col / tgt_lang_col in configs/data/config.yaml "
        "to match the actual flat string column names."
    )



def main() -> None:
    """Download the dataset and optionally inspect it.

    Parses CLI arguments and runs the download phase. The phase is idempotent;
    if the data already exists it is loaded from disk.

    Use ``--inspect`` to print dataset column names and sample rows, then exit
    without saving. This is useful to verify that
    ``src_lang_col``/``tgt_lang_col`` in the config match the actual columns
    before committing to a full pipeline run.
    """
    parser = argparse.ArgumentParser(
        description="Data preparation pipeline for the EN→ES translation project."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data/config.yaml",
        help="Path to the data configuration YAML file.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Download the dataset, print column names and samples, then exit.",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)

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
        print("Adjust src_lang_col / tgt_lang_col in configs/data/config.yaml "
              "to match the actual column names above.")
        return

    logger.info("Data pipeline complete. Raw text saved to data/raw/.")


if __name__ == "__main__":
    main()
