# Project Overview: English → Spanish Machine Translation Baseline

This repository implements a **vanilla PyTorch** baseline for English→Spanish (EN→ES) machine translation. It is intentionally written without any distributed training primitives (no Accelerate, no DDP, no DeepSpeed, no FSDP) so that students can add those optimisations themselves and measure the speedup.

---

## Table of Contents

1. [The Dataset](#1-the-dataset)
2. [Tokenisation](#2-tokenisation)
3. [Data Pipeline & DataLoader](#3-data-pipeline--dataloader)
4. [Model Architectures](#4-model-architectures)
5. [What the Models Learn to Do](#5-what-the-models-learn-to-do)
6. [Training Configuration](#6-training-configuration)
7. [Running the Project](#7-running-the-project)
8. [Project Structure](#8-project-structure)

---

## 1. The Dataset

**Source:** [`Iker/OpenHermes-2.5-English-Spanish`](https://huggingface.co/datasets/Iker/OpenHermes-2.5-English-Spanish) on Hugging Face Hub.

This is a **parallel corpus** of human–AI conversations that have been manually translated from English to Spanish. Each row stores two aligned conversation threads — one in English, one in Spanish.

### Raw Format

The dataset's underlying structure is:

```
conversations_english: [
    {"from": "human", "value": "What is the capital of France?"},
    {"from": "gpt",   "value": "The capital of France is Paris. Paris is..."},
    ...
]
conversations_spanish: [
    {"from": "human", "value": "¿Cuál es la capital de Francia?"},
    {"from": "gpt",   "value": "La capital de Francia es París. París es..."},
    ...
]
```

Turn `i` in `conversations_english` is the direct translation of turn `i` in `conversations_spanish`. The lists are fully aligned by index.

### What We Actually Use

We **only extract human-written turns** (`from = "human"`). GPT-generated responses are long, multi-paragraph answers that frequently exceed `max_seq_len = 256` tokens and get heavily truncated, which adds noise. Human turns are concise, natural sentences — exactly what a translation model should learn.

This produces a flat table of `(en, es)` string pairs, one row per human turn across all conversations.

### Dataset Size and Splits

| Split | Proportion | Approximate size |
|-------|-----------|-----------------|
| Train | 90 %      | ~810 k pairs    |
| Val   | 5 %       | ~45 k pairs     |
| Test  | 5 %       | ~45 k pairs     |

Splits are carved deterministically with `seed=42`.

---

## 2. Tokenisation

### Why a Shared Bilingual Tokeniser?

A **single BPE tokeniser** is trained on both English and Spanish text together. This is the key design choice that unlocks weight tying (see [Model Architectures](#4-model-architectures)).

English and Spanish share many sub-word units:

| English       | Spanish       | Shared sub-word |
|---------------|---------------|-----------------|
| `nation`      | `nación`      | `na`, `tion`/`ción` |
| `transform`   | `transformar` | `transform`     |
| `natural`     | `natural`     | `natural`       |

A bilingual vocabulary of 32 000 tokens covers both languages efficiently.

### BPE Algorithm

**Byte Pair Encoding (BPE)** starts with individual characters and iteratively merges the most frequent adjacent pair:

```
Corpus: "low low low lower lower newest newest"

Step 1: most frequent pair: (e, s) → "es"
        "low low low low er lower newest "es"test"
Step 2: most frequent pair: (es, t) → "est"
        ...
After N merges: "low", "lower", "new", "est" are individual vocabulary entries
```

Our tokeniser adds:
- **NFKC normalisation** — converts visually identical Unicode characters to a canonical form so `ﬁ` and `fi` receive the same ID.
- **ByteLevel pre-tokeniser** — operates on raw bytes, so emojis and any Unicode character can always be represented. No unknown characters exist.
- **TemplateProcessing** — automatically wraps every encoded sequence with `<s>` (BOS) and `</s>` (EOS) tokens.

### Tokenisation Example

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("data/processed/bpe_tokenizer.json")

# Encode a source sentence
enc = tokenizer.encode("What is the capital of France?")
print(enc.tokens)
# → ['<s>', 'What', 'Ġis', 'Ġthe', 'Ġcapital', 'Ġof', 'ĠFrance', '?', '</s>']
print(enc.ids)
# → [1, 423, 278, 264, 4891, 309, 8731, 35, 2]

# Encode the corresponding target sentence
enc_es = tokenizer.encode("¿Cuál es la capital de Francia?")
print(enc_es.tokens)
# → ['<s>', 'ĠCuál', 'Ġes', 'Ġla', 'Ġcapital', 'Ġde', 'ĠFrancia', '?', '</s>']
print(enc_es.ids)
# → [1, 5042, 391, 273, 4891, 284, 9214, 35, 2]

# Decode back to text
print(tokenizer.decode(enc.ids))
# → "<s> What is the capital of France? </s>"
```

Note that `Ġ` (Unicode 0x0120) is BPE's representation of a leading space — it distinguishes the word-initial use of a sub-word from a word-internal use.

### Special Tokens

| Token   | ID | Role                                               |
|---------|----|----------------------------------------------------|
| `<pad>` | 0  | Padding — ignored in attention and in the loss     |
| `<s>`   | 1  | Begin-of-Sequence — first decoder input token      |
| `</s>`  | 2  | End-of-Sequence — target for last generated token  |
| `<unk>` | 3  | Unknown — never used in practice (ByteLevel covers all) |

---

## 3. Data Pipeline & DataLoader

The pipeline (`src/text.py`) runs three idempotent phases:

```
Phase 1 — download_dataset()
    Downloads the raw dataset from Hugging Face Hub, extracts human turns,
    and saves split Arrow files to data/raw/.

Phase 2 — build_tokenizer()
    Trains BPE on the training split and saves data/processed/bpe_tokenizer.json.

Phase 3 — tokenize_and_save()
    Tokenises every split, truncates to max_seq_len=256, and saves
    integer-ID Arrow files to data/processed/{train,val,test}/.
    Padding is NOT applied here — it happens dynamically per batch.
```

### Dynamic Padding and BucketSampler

Naïve training pads every sequence to the global maximum length (256). At `batch_size=32`:

```
Naïve:    32 × 256 = 8 192 positions — but average sentence is ~40 tokens
          → ~84% of computation is wasted on padding tokens
```

This project uses two techniques to eliminate that waste:

**1. BucketSampler** — sorts sequences by length and groups similar-length sequences into the same batch. Within a batch of short sentences (20–30 tokens), no sequence needs padding beyond 30.

**2. Dynamic padding in `collate_fn`** — each batch is padded only to the length of its own longest sequence, not the global maximum.

```
With BucketSampler + dynamic padding:
  Short batch: 32 × 28 = 896 positions   (90% reduction)
  Long batch:  32 × 180 = 5 760 positions (still better than 8 192)
  Average padding waste: < 5%            (~5× more real tokens/sec)
```

The padding mask convention is `True = padding position, False = real token` — compatible with PyTorch's `nn.MultiheadAttention`.

---

## 4. Model Architectures

Two architectures are provided, both with the same interface (`forward` returns logits, `generate` does greedy decoding):

---

### Option A — Transformer from Scratch (`src/model.py`)

A complete reimplementation of *Attention is All You Need* (Vaswani et al., 2017) with Pre-LayerNorm.

#### Architecture Diagram

```
Input IDs [B, src_len]
    │
Embedding  (vocab_size × d_model, shared for src/tgt)
    │  × √d_model  (scaling)
    │
PositionalEncoding  (sinusoidal, non-trainable buffer)
    │
┌───┴────────────────────────────────────────────────┐
│  TransformerEncoder  (N × EncoderLayer)             │
│                                                     │
│  EncoderLayer:                                      │
│    LayerNorm → Multi-Head Self-Attention → residual │
│    LayerNorm → Feed-Forward Network      → residual │
└───┬────────────────────────────────────────────────┘
    │ memory [B, src_len, d_model]
    │
┌───┴────────────────────────────────────────────────┐
│  TransformerDecoder  (N × DecoderLayer)             │
│                                                     │
│  DecoderLayer:                                      │
│    LayerNorm → Masked Self-Attention     → residual │
│    LayerNorm → Cross-Attention (↑memory) → residual │
│    LayerNorm → Feed-Forward Network      → residual │
└───┬────────────────────────────────────────────────┘
    │
Output Projection  (weight-tied with tgt embedding)
    │
Logits [B, tgt_len, vocab_size]
```

#### Multi-Head Attention

Each attention head independently computes:

```
Attention(Q, K, V) = softmax( Q @ Kᵀ / √head_dim ) @ V
```

Where:
- `Q = W_q · x`,  `K = W_k · x`,  `V = W_v · x`  (learned projections)
- `head_dim = d_model / nhead`
- Division by `√head_dim` prevents dot products from growing too large and pushing softmax into saturation (near-zero gradients)

The outputs of all heads are concatenated and projected back: `output = W_o · concat(heads)`.

The decoder uses **two** attention sub-layers:
- **Masked self-attention** — causal mask (upper-triangular `-inf`) prevents each position from attending to future tokens.
- **Cross-attention** — queries come from the decoder, keys/values come from the encoder output. This is the "reading" step where the decoder consults the source sentence.

#### Pre-LayerNorm vs Post-LayerNorm

The original paper uses **Post-LN**: `x = LayerNorm(x + Sublayer(x))`.

This project uses **Pre-LN** (`norm_first=True`): `x = x + Sublayer(LayerNorm(x))`.

```
Post-LN:  LN sits on the residual path → gradients decay exponentially with depth
Pre-LN:   LN is inside the branch      → residual stream stays clean, gradients
                                          flow freely through skip connections
```

Pre-LN enables reliable training of deep models (8, 12, 24+ layers) without special initialisation tricks.

#### Weight Tying

Because source and target share the same tokeniser, they share the same embedding matrix. The output projection reuses those weights transposed:

```python
logit_for_token_t = dot(hidden_state, embedding_vector_of_t)
```

This enforces a consistent vector space (an inductive bias: a token's embedding and its prediction direction are the same vector) and saves `vocab_size × d_model` parameters.

#### Model Presets

| Preset  | `d_model` | `nhead` | Layers (enc+dec) | `dim_ff` | Parameters |
|---------|-----------|---------|------------------|----------|-----------|
| SMALL   | 512       | 8       | 6 + 6            | 2 048    | ~61 M     |
| FULL    | 1 024     | 16      | 6 + 6            | 4 096    | ~205 M    |

Parameter breakdown for SMALL:

```
Embedding (shared):         32 000 × 512 = 16 M
Each EncoderLayer:
  MHA (W_q, W_k, W_v, W_o): 4 × 512² ≈ 1.0 M
  FFN (two linears):         2 × 512 × 2048 ≈ 2.0 M
  Total per layer:           ≈ 3 M
Each DecoderLayer:
  Self-Attn + Cross-Attn + FFN: ≈ 4.5 M
Output projection:          weight-tied → 0 extra params

Total SMALL ≈ 16 M + 6×3 M + 6×4.5 M = 61 M params
```

---

### Option B — Pre-trained Fine-tuning Wrapper (`src/model_pretrained.py`)

Wraps [`Helsinki-NLP/opus-mt-en-es`](https://huggingface.co/Helsinki-NLP/opus-mt-en-es) — a MarianMT model (~74 M parameters) pre-trained on large EN→ES parallel corpora.

```python
class PretrainedTranslationModel(nn.Module):
    def __init__(self, model_name="Helsinki-NLP/opus-mt-en-es"):
        self.model = MarianMTModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             labels=labels)
        return outputs.loss, outputs.logits
```

**Important:** Option B uses `MarianTokenizer` (the model's own vocabulary), not the shared BPE tokeniser from Option A. The two vocabularies are incompatible.

| Property              | Option A (from scratch) | Option B (fine-tuning)  |
|-----------------------|------------------------|------------------------|
| Starting point        | Random weights         | Pre-trained EN→ES weights |
| Epochs to useful BLEU | 5–10                   | 1–2                    |
| Learning rate         | 1×10⁻⁴                 | 5×10⁻⁵ (lower to avoid catastrophic forgetting) |
| Warmup steps          | 4 000                  | 500                    |
| Use case              | Understand training from scratch; benchmark hardware optimisations | Fast sanity-check; compare fine-tuning efficiency |

---

## 5. What the Models Learn to Do

Both models solve the **sequence-to-sequence translation** task: given a sequence of English token IDs, produce a sequence of Spanish token IDs.

### During Training — Teacher Forcing

At training time the decoder receives the **ground-truth** target tokens shifted one position to the right. At each position `i`, the decoder predicts token `i+1` given the correct tokens `0..i`:

```
Source (encoder input):    <s>  What  is  the  capital  of  France?  </s>
                            ↓    ↓     ↓    ↓     ↓       ↓    ↓       ↓
                           [ encoder processes all positions in parallel ]
                                        ↓ memory
Target decoder input:      <s>  ¿Cuál  es  la  capital  de  Francia?
Target labels (loss):      ¿Cuál  es   la  capital  de  Francia?  </s>
```

The loss is **cross-entropy with label smoothing (ε=0.1)**. Instead of a hard one-hot target, each position's distribution is softened:

```
Hard target:    [0, 0, ..., 1, ..., 0]          (100% on correct token)
Smoothed target:[ε/V, ..., 1−ε+ε/V, ..., ε/V]  (small probability mass on all tokens)
```

Label smoothing prevents overconfidence and improves BLEU.

### At Inference — Greedy Decoding

At evaluation time the model generates autoregressively:

```python
generated = [<s>]
for step in range(max_new_tokens):
    logits = model(src, generated)          # forward pass
    next_token = logits[:, -1, :].argmax()  # pick highest-probability token
    generated.append(next_token)
    if next_token == </s>:
        break
```

```
Step 0: input=[<s>]                   → predict: ¿Cuál
Step 1: input=[<s>, ¿Cuál]            → predict: es
Step 2: input=[<s>, ¿Cuál, es]        → predict: la
Step 3: input=[<s>, ¿Cuál, es, la]    → predict: capital
...
Step N: input=[..., Francia, ?]        → predict: </s>  → STOP

Output: "¿Cuál es la capital de Francia?"
```

### Evaluation Metric — BLEU

Translation quality is measured with **BLEU** (Bilingual Evaluation Understudy). BLEU counts n-gram overlaps between the model output and reference translations, normalised by output length. It ranges from 0 (no overlap) to 100 (perfect match).

| BLEU score | Qualitative interpretation     |
|-----------|-------------------------------|
| < 10      | Almost useless                |
| 10–19     | Hard to understand            |
| 20–29     | Some essence captured         |
| 30–40     | Understandable, decent quality|
| > 40      | High quality (near human)     |

Expected results:
- Option A (from scratch, 10 epochs): BLEU ~15–25 depending on hardware and batch size
- Option B (fine-tuned, 2–3 epochs): BLEU > 20 from epoch 1

---

## 6. Training Configuration

### Optimiser — AdamW + Noam Schedule

Both training scripts use **AdamW** with the Noam learning-rate schedule from the original Transformer paper:

```
lr = d_model^(-0.5) × min(step^(-0.5), step × warmup_steps^(-1.5))
```

This linearly increases the learning rate for `warmup_steps` steps, then decays it proportionally to the inverse square root of the step. The schedule prevents instability in the first steps (when gradients are large and weights are miscalibrated) and ensures the learning rate falls as the model converges.

```
                 warmup_steps
                      │
LR   ▲                │
     │       /─────── │
     │      /         │ ──────────────────
     │     /          │
     └─────────────────────────────► step
```

**`beta2=0.98`** (vs PyTorch default `0.999`) gives the second-moment estimate a shorter memory, allowing faster adaptation to changing gradient magnitudes in the early training steps of a Transformer.

### Gradient Clipping

The global gradient norm is clipped to `max_grad_norm=1.0` before each optimiser step. This prevents exploding gradients (especially in the first epochs when attention weights are poorly calibrated) without changing the gradient direction.

### Throughput Metric

The primary throughput metric is **tokens per second** — the number of non-padding source and target tokens processed per second:

```python
n_tokens = (src_ids != pad_id).sum() + (tgt_ids != pad_id).sum()
tokens_per_sec = n_tokens / batch_time
```

This is the correct metric for comparing hardware configurations because padding tokens cost compute but carry no information.

---

## 7. Running the Project

```bash
# Install dependencies
pip install -r requirements.txt

# Step 1: inspect dataset columns (recommended first run)
python src/text.py --config configs/data/config.yaml --inspect

# Step 2: run full data pipeline (download → tokenise → save)
python src/text.py --config configs/data/config.yaml

# Step 3a: train from scratch (Option A) — small preset
python scripts/train.py \
    --data-config configs/data/config.yaml \
    --train-config configs/train/text_small.yaml

# Step 3a: train from scratch (Option A) — full preset
python scripts/train.py \
    --data-config configs/data/config.yaml \
    --train-config configs/train/text_full.yaml

# Step 3b: fine-tune pre-trained model (Option B)
python scripts/train_pretrained.py \
    --data-config configs/data/config.yaml \
    --train-config configs/train/text_pretrained.yaml

# Resume from a checkpoint
python scripts/train.py \
    --data-config configs/data/config.yaml \
    --train-config configs/train/text_small.yaml \
    --resume models/checkpoint_epoch2.pt
```

---

## 8. Project Structure

```
infradeep/
├── configs/
│   ├── data/
│   │   └── config.yaml              # Dataset, tokeniser, max_seq_len, split ratios
│   └── train/
│       ├── text_small.yaml          # Option A, ~61M params (d_model=512)
│       ├── text_full.yaml           # Option A, ~205M params (d_model=1024)
│       └── text_pretrained.yaml     # Option B, fine-tuning hyperparameters
│
├── src/
│   ├── text.py                      # Data pipeline: download → BPE → tokenise
│   ├── dataset.py                   # TranslationDataset, BucketSampler, collate_fn
│   ├── model.py                     # Transformer from scratch (Option A)
│   ├── model_pretrained.py          # MarianMT fine-tuning wrapper (Option B)
│   ├── utils.py                     # Logger, checkpointing, BLEU, Noam scheduler
│   └── __init__.py                  # Public re-exports
│
├── scripts/
│   ├── train.py                     # Training loop for Option A
│   └── train_pretrained.py          # Training loop for Option B
│
├── data/                            # Created at runtime
│   ├── raw/                         # Downloaded Arrow files
│   └── processed/                   # Tokenised Arrow files + BPE tokeniser JSON
│
├── models/                          # Checkpoints saved here during training
├── logs/                            # Training logs
├── docs/
│   └── project_overview.md          # This file
└── requirements.txt
```
