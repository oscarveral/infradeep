"""Inference and evaluation script for the EN→ES translation model (Option B).

Uses the fine-tuned MarianMT wrapper. Three operating modes are available,
and they can be combined freely in a single call:

- **Sentence mode** (``--sentences``): translate one or more strings passed
  directly on the command line.
- **Interactive mode** (``--interactive``): start a read-eval-print loop that
  accepts sentences from stdin one at a time.
- **Test-set mode** (``--test-set``): run greedy decoding over the full test
  split and report corpus-level BLEU.

Usage examples::

    # Option B — translate sentences (no fine-tuning, just base pre-trained)
    python scripts/test.py --sentences "Good morning!"

    # Option B — evaluate fine-tuned checkpoint on test set
    python scripts/test.py \\
        --checkpoint models/checkpoint_ep003_step0012500.pt \\
        --data-config configs/data/config.yaml \\
        --test-set

    # Combine sentence + interactive modes
    python scripts/test.py \\
        --checkpoint models/checkpoint_ep003_step0012500.pt \\
        --data-config configs/data/config.yaml \\
        --sentences "The sky is blue." \\
        --interactive
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import compute_bleu, count_parameters, create_logger, load_yaml_config

logger = create_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Default example sentences used when no --sentences / --interactive /
# --test-set flags are provided (demo mode).
# ──────────────────────────────────────────────────────────────────────────────
_DEMO_SENTENCES = [
    "Hello, how are you?",
    "The capital of France is Paris.",
    "Machine learning is a branch of artificial intelligence.",
    "I would like to book a table for two people.",
    "What time does the next train leave?",
    "She has been studying Spanish for three years.",
    "The weather today is sunny and warm.",
    "Could you please help me with my luggage?",
]


# ══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Inference and evaluation for EN→ES translation models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Model selection ──────────────────────────────────────────────────────
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a .pt checkpoint saved by train_pretrained.py.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Helsinki-NLP/opus-mt-en-es",
        help="HuggingFace model name (default: Helsinki-NLP/opus-mt-en-es).",
    )

    # ── Config files ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--data-config",
        type=str,
        default="configs/data/config.yaml",
        help="Path to configs/data/config.yaml (needed for test split).",
    )

    # ── Operating modes ──────────────────────────────────────────────────────
    parser.add_argument(
        "--sentences",
        nargs="+",
        metavar="SENTENCE",
        default=None,
        help="One or more English sentences to translate.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start an interactive REPL. Type a sentence and press Enter to translate.",
    )
    parser.add_argument(
        "--test-set",
        action="store_true",
        help="Evaluate corpus BLEU on the test split from data/processed/.",
    )

    # ── Generation settings ──────────────────────────────────────────────────
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate per sentence (default: 100).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for test-set evaluation (default: 32).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override: 'cpu', 'cuda', 'cuda:0', etc. Auto-detected if omitted.",
    )

    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Option B helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_option_b(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple:
    """Load the MarianMT fine-tuning wrapper and its tokeniser.

    If no checkpoint is provided, the base pre-trained model weights are used
    directly (useful for a zero-shot baseline comparison).

    Args:
        args (argparse.Namespace): Parsed CLI arguments.
        device (torch.device): Target device.

    Returns:
        tuple: ``(model, marian_tokenizer)`` where model is in eval mode.
    """
    from transformers import MarianTokenizer

    from src.model_pretrained import PretrainedTranslationModel

    logger.info(f"Loading MarianTokenizer for '{args.model_name}' …")
    marian_tokenizer = MarianTokenizer.from_pretrained(args.model_name)

    model = PretrainedTranslationModel(model_name=args.model_name)

    if args.checkpoint is not None:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        meta_epoch = state.get("epoch", "?")
        meta_step  = state.get("step", "?")
        meta_loss  = state.get("loss", float("nan"))
        logger.info(
            f"Fine-tuned checkpoint loaded: epoch={meta_epoch}, step={meta_step}, "
            f"val_loss={meta_loss:.4f}"
        )
    else:
        logger.info("No checkpoint provided — using base pre-trained weights.")

    model = model.to(device)
    model.eval()
    count_parameters(model)

    return model, marian_tokenizer


@torch.no_grad()
def _translate_option_b(
    sentences: list[str],
    model,
    marian_tokenizer,
    device: torch.device,
    max_new_tokens: int,
) -> list[str]:
    """Translate a list of English sentences with the Option B model.

    Uses ``MarianTokenizer`` for encoding and decoding. The tokenizer handles
    padding and attention masks automatically.

    Args:
        sentences (list[str]): English source strings.
        model: ``PretrainedTranslationModel`` in eval mode.
        marian_tokenizer: ``MarianTokenizer`` instance.
        device (torch.device): Model device.
        max_new_tokens (int): Maximum tokens to generate.

    Returns:
        list[str]: Translated Spanish strings.
    """
    encoding = marian_tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
    )  # [B, generated_len]

    # skip_special_tokens=True strips BOS / EOS / PAD automatically.
    return marian_tokenizer.batch_decode(generated, skip_special_tokens=True)


@torch.no_grad()
def _evaluate_test_set_option_b(
    model,
    marian_tokenizer,
    cfg_data: dict,
    device: torch.device,
    batch_size: int,
    max_new_tokens: int,
) -> float:
    """Run greedy decoding on the test split and return corpus BLEU (Option B).

    Loads raw text from the Arrow test split (not the pre-tokenised IDs), so
    that ``MarianTokenizer`` can tokenise it with its own vocabulary.

    Args:
        model: ``PretrainedTranslationModel`` in eval mode.
        marian_tokenizer: ``MarianTokenizer`` instance.
        cfg_data (dict): Data config dict.
        device (torch.device): Model device.
        batch_size (int): Sentences per batch.
        max_new_tokens (int): Maximum tokens to generate.

    Returns:
        float: Corpus BLEU score (SacreBLEU, 0–100).
    """
    import datasets as hf_datasets

    raw_dir = cfg_data["raw_data_dir"]
    if not os.path.exists(os.path.join(raw_dir, "dataset_dict.json")):
        logger.error(
            f"Raw dataset not found at '{raw_dir}'. "
            "Run the data pipeline first: python src/text.py ..."
        )
        sys.exit(1)

    dataset_dict = hf_datasets.load_from_disk(raw_dir)
    test_split = dataset_dict["test"]

    src_col = cfg_data["src_lang_col"]
    tgt_col = cfg_data["tgt_lang_col"]

    src_texts = test_split[src_col]
    tgt_texts = test_split[tgt_col]
    n = len(src_texts)

    hypotheses: list[str] = []
    references: list[list[str]] = []

    for start in tqdm(range(0, n, batch_size), desc="  Test-set decoding"):
        batch_src = src_texts[start : start + batch_size]
        batch_tgt = tgt_texts[start : start + batch_size]

        preds = _translate_option_b(
            batch_src, model, marian_tokenizer, device, max_new_tokens
        )
        hypotheses.extend(preds)
        references.extend([[t] for t in batch_tgt])

    return compute_bleu(hypotheses, references)


# ══════════════════════════════════════════════════════════════════════════════
# Pretty-print helpers
# ══════════════════════════════════════════════════════════════════════════════

def _print_translations(sentences: list[str], translations: list[str]) -> None:
    """Print source/translation pairs in a readable table.

    Args:
        sentences (list[str]): English source strings.
        translations (list[str]): Spanish translations.
    """
    width = 70
    print("\n" + "═" * width)
    print(f"  {'EN (source)':<34}  {'ES (translation)'}")
    print("═" * width)
    for src, tgt in zip(sentences, translations):
        # Wrap long strings for readability.
        src_disp = src[:60] + "…" if len(src) > 60 else src
        tgt_disp = tgt[:60] + "…" if len(tgt) > 60 else tgt
        print(f"  {src_disp:<34}  {tgt_disp}")
    print("═" * width + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Orchestrate inference and evaluation for the EN→ES translation models.

    Reads CLI flags, loads the appropriate model, and runs the selected modes
    in order: sentences → interactive → test-set. If none of the three modes
    are requested, falls back to a set of built-in demo sentences.
    """
    args = parse_args()

    # ── Device ───────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # ── Load model ───────────────────────────────────────────────────────────
    cfg_data = load_yaml_config(args.data_config)
    model, marian_tokenizer = _load_option_b(args, device)

    def translate(sentences: list[str]) -> list[str]:
        return _translate_option_b(
            sentences, model, marian_tokenizer, device, args.max_new_tokens
        )

    def evaluate_test_set() -> float:
        return _evaluate_test_set_option_b(
            model, marian_tokenizer, cfg_data, device, args.batch_size, args.max_new_tokens
        )

    # ── Determine which modes to run ─────────────────────────────────────────
    any_mode = args.sentences or args.interactive or args.test_set

    # ── Mode 1: custom sentences ─────────────────────────────────────────────
    if args.sentences:
        logger.info(f"Translating {len(args.sentences)} custom sentence(s) …")
        translations = translate(args.sentences)
        _print_translations(args.sentences, translations)

    # ── Mode 2: demo sentences (only if no explicit mode is requested) ────────
    if not any_mode:
        logger.info("No mode specified — running built-in demo sentences.")
        translations = translate(_DEMO_SENTENCES)
        _print_translations(_DEMO_SENTENCES, translations)

    # ── Mode 3: interactive REPL ─────────────────────────────────────────────
    if args.interactive:
        print("\nInteractive mode — type an English sentence and press Enter.")
        print("Type 'quit' or press Ctrl-C to exit.\n")
        try:
            while True:
                try:
                    sentence = input("EN > ").strip()
                except EOFError:
                    break
                if not sentence:
                    continue
                if sentence.lower() in ("quit", "exit", "q"):
                    break
                translation = translate([sentence])[0]
                print(f"ES > {translation}\n")
        except KeyboardInterrupt:
            pass
        print("Exiting interactive mode.")

    # ── Mode 4: test-set BLEU ─────────────────────────────────────────────────
    if args.test_set:
        logger.info("Evaluating on test set (greedy decoding) …")
        bleu = evaluate_test_set()
        print("\n" + "═" * 40)
        print(f"  Test-set BLEU (SacreBLEU): {bleu:.2f}")
        print("═" * 40 + "\n")


if __name__ == "__main__":
    main()
