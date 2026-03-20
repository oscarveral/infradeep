"""Pre-trained model wrapper for fine-tuning (Option B).

Wraps the "Helsinki-NLP/opus-mt-en-es" MarianMT model (~74 M parameters)
from HuggingFace Transformers in a plain "nn.Module" interface.

Option B exists because training a 65-220 M parameter Transformer from scratch
(Option A) requires large amounts of data and compute to converge to useful
translations. Option B fine-tunes a model that already understands language and
translation, reaching BLEU > 20 within 1-2 epochs. Students can use it to:

1. Debug the training pipeline quickly — if BLEU does not improve after one
   fine-tuning epoch there is a bug in the loop, not in the model.
2. Compare fine-tuning vs from-scratch training efficiency.
3. Apply the same hardware optimisations (DDP, DeepSpeed, FSDP) to a
   pre-trained model and benchmark the speedup.

The HuggingFace "Trainer" class is **not** used. The optimiser, loss
computation, and backward pass all live in "scripts/train_pretrained.py",
exactly as in the Option A training script. This keeps the learning objective
clear: students see exactly where every gradient comes from.

Option B uses "MarianTokenizer" (loaded in "scripts/train_pretrained.py"),
not the custom BPE tokeniser from "src/text.py". The pre-trained model was
trained with its own vocabulary and cannot be used with a different tokeniser.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from transformers import MarianMTModel


class PretrainedTranslationModel(nn.Module):
    """Thin "nn.Module" wrapper around "MarianMTModel".

    Exposes a "forward()" that returns "(loss, logits)" and a "generate()"
    for greedy decoding — the same interface as the from-scratch model in
    "scripts/train.py" — so both training scripts share the same structure.

    Args:
        model_name (str, optional): HuggingFace Hub identifier for the
            pre-trained model. Defaults to
            ""Helsinki-NLP/opus-mt-en-es"" (~74 M parameters). Alternative:
            ""Helsinki-NLP/opus-mt-tc-big-en-es"" (~230 M).
    """

    def __init__(
        self,
        model_name: str = "Helsinki-NLP/opus-mt-en-es",
    ) -> None:
        super().__init__()

        self.model = MarianMTModel.from_pretrained(model_name)
        self.model_name = model_name

        self.vocab_size = self.model.config.vocab_size
        self.pad_token_id = self.model.config.pad_token_id

        # The pre-trained GenerationConfig ships with max_length=512.  If we
        # also pass max_new_tokens to generate(), HuggingFace warns that both
        # are set and picks max_new_tokens.  Clearing max_length here removes
        # the conflict so the warning never appears.
        self.model.generation_config.max_length = None

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
        label_smoothing: float = 0.0,
    ) -> tuple[Tensor, Tensor]:
        """Compute the forward pass and return "(loss, logits)".

        "MarianMTModel" internally shifts "labels" right to build decoder
        inputs (teacher-forcing), equivalent to Option A's manual
        "tgt[:, :-1]" / "tgt[:, 1:]" split. Logits are returned alongside the
        loss so the training script can log perplexity.

        When "label_smoothing > 0" the internal HuggingFace loss is replaced
        with a custom "nn.CrossEntropyLoss" computed from the logits. This
        ensures the "label_smoothing" field in "text_pretrained.yaml" is
        actually applied; without this the config value was silently ignored
        because "MarianMTModel" computes plain cross-entropy internally.

        Label smoothing converts the hard one-hot target distribution
        "[0, …, 1, …, 0]" into a soft distribution
        "[(ε/V), …, (1 − ε + ε/V), …, (ε/V)]" (ε = label_smoothing,
        V = vocab_size). This prevents the model from becoming too confident
        and improves generalisation / BLEU.

        Args:
            input_ids (Tensor): Source token IDs, shape "[batch, src_len]".
            attention_mask (Tensor): Source padding mask "[batch, src_len]",
                "1" = real token, "0" = padding.
            labels (Tensor): Target token IDs "[batch, tgt_len]". Padding
                positions must be "-100" (HuggingFace ignore_index convention).
            label_smoothing (float, optional): Smoothing factor ε in [0, 1).
                "0.0" uses the model's built-in CE loss. Defaults to "0.0".

        Returns:
            tuple[Tensor, Tensor]:
                - **loss** (Tensor): Scalar cross-entropy loss, mean over
                  non-padding positions.
                - **logits** (Tensor): Unnormalised log-probabilities,
                  shape "[batch, tgt_len, vocab_size]".
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        if label_smoothing > 0.0:
            # Replace the model's plain CE with smoothed CE so the config
            # field label_smoothing is honoured.  The logits already have the
            # correct shape [B, T, V]; we flatten batch × time for the loss.
            criterion = nn.CrossEntropyLoss(
                ignore_index=-100,
                label_smoothing=label_smoothing,
            )
            loss = criterion(
                outputs.logits.reshape(-1, outputs.logits.size(-1)),  # [B*T, V]
                labels.reshape(-1),                                    # [B*T]
            )
        else:
            loss = outputs.loss

        return loss, outputs.logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        max_new_tokens: int = 100,
        num_beams: int = 1,
    ) -> Tensor:
        """Greedy (or beam-search) decoding for BLEU evaluation.

        Args:
            input_ids (Tensor): Source token IDs, shape "[batch, src_len]".
            attention_mask (Tensor): Source padding mask, shape
                "[batch, src_len]".
            max_new_tokens (int, optional): Maximum number of tokens to
                generate. Defaults to "100".
            num_beams (int, optional): "1" = greedy decoding; ">1" = beam
                search. Greedy is faster and sufficient for BLEU estimation.
                Defaults to "1".

        Returns:
            Tensor: Generated token IDs, shape
                "[batch, generated_len]", including BOS/EOS tokens.
        """
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            # Prevent repetition loops.  Without these controls, greedy
            # decoding can get stuck emitting the same token indefinitely
            # once the decoder context drifts into a high-probability region
            # for a common token (e.g. "e", "os").
            #
            # no_repeat_ngram_size=3: any 3-gram that has already appeared
            # in the output is blocked from appearing again.
            # repetition_penalty=1.3: logit of any already-seen token is
            # divided by 1.3, reducing its probability multiplicatively.
            no_repeat_ngram_size=3,
            repetition_penalty=1.3,
        )

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return (
            f"PretrainedTranslationModel(\n"
            f"  model_name={self.model_name!r},\n"
            f"  n_params={n_params:,},\n"
            f"  vocab_size={self.vocab_size}\n"
            f")"
        )
