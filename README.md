# Modular Tokenization

This repository contains the code required to reproduce the experiments from the paper
**_To Each Language Its Tokenizer: Modular Tokenizers for Efficient Multilingual LLMs_** (EMNLP 2026).

Multilingual LLMs traditionally share a single vocabulary across all supported languages, which
leads to uneven compression across them; their large embedding and output matrices increase
memory usage and slow inference, notably for small models — wastefully so, since a model is
often used for only a subset of its languages.

This repository implements a modular framework for multilingual model training that addresses
both issues:

- **Modular tokenizers** (`modular_tokenizers`): methods to learn large modular BPE and Unigram
  tokenizers from which a **subtokenizer** tailored to any language subset can be extracted.
  These subtokenizers achieve compression on par with monolingual tokenizers and improve
  cross-lingual fairness.
- **Modular pretraining and inference** (`modular_lm`): a pretraining strategy that samples a
  subtokenizer for each batch and restricts predictions to its vocabulary subset, training
  efficiently despite the large global vocabulary. The resulting models run inference with any
  combination of language-specific vocabularies — less memory and faster inference, without
  sacrificing performance.

---

## 🗂️ Table of Contents
- [Installation](#installation)
- [Training Tokenizers](#training-tokenizers)
- [Training Language Models](#training-language-models)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Folder Structure](#folder-structure)
- [Licenses](#licenses)

---

## Installation

### 1️⃣ Clone this repository
```sh
git clone git@github.com:kyutai-labs/modular-tokenization.git
cd modular-tokenization
```

### 2️⃣ Install dependencies

We recommend using [`uv`](https://docs.astral.sh/uv/) to manage the environment.
It automatically resolves dependencies from `pyproject.toml`: simply prefix every
command with `uv run`.

```sh
uv sync
```

This installs the whole project, including the language-model stack (JAX with CUDA,
flash attention, metrics — Linux only). To use only the tokenizer package
(`modular_tokenizers`, no GPU dependencies, runs anywhere):

```sh
uv sync --no-group lm
```

### Installing without `uv`

With `pip` (≥ 25.1) you will need **Python ≥ 3.11**:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e . --group lm     # drop "--group lm" for the tokenizer package only
```

---

## Training Tokenizers

Every command that reads training text supports the following sources: plain text (one document
per line), JSONL (`text_field=...`), the `.zst`-compressed version of either, or a directory of
such files. All paths are explicit arguments:

```bash
sbatch scripts/tokenizers/train_monolingual_unigram.sh   # per-language Unigram (HF or SentencePiece)
sbatch scripts/tokenizers/merge_unigram.sh               # merged Unigram + EM re-estimation (paper §3.1)
sbatch scripts/tokenizers/train_monolingual_bpe.sh       # per-language BPE references
sbatch scripts/tokenizers/train_sequential_bpe.sh        # sequential BPE (paper §3.2, Algorithm 1)
sbatch scripts/tokenizers/eval_nsl.sh                    # NSL vs monolingual references (paper Eq. 1)
```

Every script declares its inputs in an `edit me` block at the top — edit it in place, or
override any variable through the environment without touching the file:

```bash
sbatch --export=ALL,VOCAB_SIZE=16000,OUT=outputs/tokenizers \
  scripts/tokenizers/train_sequential_bpe.sh
```

(`eval_nsl.sh` takes its per-language paths from a YAML file instead: copy
`scripts/tokenizers/nsl.example.yaml` and edit it.)

Training writes a **tokenizer directory** — the model file (`tokenizer.json` or
`tokenizer.model`) next to `lang_to_ids.json` (the per-language global token ids) — which is
everything `ModularTokenizer` needs to extract subtokenizers:

```python
from modular_tokenizers import TokenizerArgs

tokenizer = TokenizerArgs(
    type="modular",
    path="outputs/tokenizers/sequential_bpe_24000",
    langs="en,fr,de",
    extraction_strategy={"strategy": "merged_seq_bpe"},   # merged_norm for unigram
).build_tokenizer()

ids = tokenizer.encode("Bonjour le monde", "fr")   # global ids, compact fr sub-vocabulary
tokenizer.decode(ids, "fr")
tokenizer.mask("fr")                               # boolean vocabulary mask (for the LM loss)
```

Note: the EM re-estimation step is pure Python and scales poorly with corpus size — run it on a
small sample with capped iterations (`max_input_sentences=10000`, `max_iters=50`, as the script
does): this is sufficient to reproduce the paper's compression results.

---

## Training Language Models

```bash
sbatch scripts/lm/train.sh
```

All `scripts/lm/` launchers follow the same convention as the tokenizer ones: edit the
`edit me` block, or override its variables via `sbatch --export`. The underlying command
looks like:

```bash
uv run python -m modular_lm.train \
  run_dir=outputs/models/run_0 \
  data.sources.en=/path/to/en data.weights.en=0.5 \
  data.sources.fr=/path/to/fr data.weights.fr=0.5 \
  data.batch_size=8 data.context_size=2048 \
  model.dim=1024 model.n_layers=24 \
  optim.total_steps=50000 \
  tokenizer.type=modular \
  tokenizer.path=outputs/tokenizers/sequential_bpe_24000 \
  tokenizer.langs=en,fr \
  'tokenizer.sampling_strategy={strategy: uniform, p_lang: 0.5, n_extra_langs: 2, n_sampled_subtokenizers: 10}' \
  'tokenizer.extraction_strategy={strategy: merged_seq_bpe}'
```

One SLURM task per node owns all its GPUs. Batches are monolingual; each document is tokenized
by a subtokenizer sampled from the tokenizer's `sampling_strategy`, and the loss is masked to
that sub-vocabulary (scaled to bits-per-byte, so losses are comparable across subtokenizers).

> **Note**: in this version the restriction to the sub-vocabulary is implemented with an
> additive logits mask over the full output matrix — the CPU offloading described in the paper
> (keeping the full embedding/output matrices in host memory and materializing only the sampled
> sub-vocabulary rows on device) is **not** implemented here.

Checkpoints are one orbax composite per step (`params`, `opt_state`, `meta`) plus a ~1KB
per-rank dataloader snapshot. To continue a run, pass `resume=true` (and optionally
`resume_step=<step>` — later checkpoints are then discarded): the data stream resumes
**bit-exactly**, provided the world size and Python version are unchanged. A fresh run refuses
a non-empty `run_dir`. When resuming a cosine run, keep `optim.total_steps` identical (it
shapes the schedule); the `wsd` scheduler is designed for open-ended extension.

---

## Evaluation

```bash
sbatch scripts/lm/eval_mcq.sh             # ARC, CSQA, PIQA, SIQA, HellaSwag, MMLU, ... (cloze or letter)
sbatch scripts/lm/eval_translation.sh     # FLORES-style parallel data, BLEU (+ EM/F1)
sbatch scripts/lm/eval_summarization.sh   # XLSum / EUR-Lex-Sum / WikiLingua, ROUGE
sbatch scripts/lm/latency.sh              # forward latency: compact sub-vocabulary vs full vocabulary
```

Each evaluation takes an explicit `data_path` pointing at one task's own directory
(`train.jsonl` + `test.jsonl` in the dataset's original format — the expected fields are
documented in each module's docstring), and a `subtokenizer_ids` list to evaluate under.
Translation supports a cross-subtokenizer mode (`subtokenizer_id_out`) where the few-shot
prompt is built in token space, sources and targets encoded by different subtokenizers.

Evaluation loads checkpoints trained with this code as well as the paper's original runs
(both legacy layouts), restoring only `params`.

---

## Inference

```bash
# interactive, or prompts=file.txt (one prompt per line)
uv run python -m modular_lm.inference.generate \
  run_dir=outputs/models/run_0 \
  subtokenizer_id=fr \
  length=64
```

Generation encodes the prompt with `subtokenizer_id` and restricts sampling to
`subtokenizer_id_out` (default: the same) through a logits mask; token ids stay global, so any
subtokenizer can continue another one's prompt.

---

## Folder Structure

```
modular_tokenizers/        # building & using modular tokenizers (no GPU deps)
 ├── training/
 │    ├── bpe/             # sequential BPE (paper §3.2) + serialization
 │    └── unigram/         # merged Unigram: train, merge, EM re-estimation (paper §3.1)
 ├── subtokenizers/        # ModularTokenizer: masks, extraction, sampling
 ├── evaluation/           # NSL (paper Eq. 1)
 └── config.py             # config parsing shared by all CLIs (YAML + dotted overrides)

modular_lm/                # JAX language modeling on modular tokenizers
 ├── train.py              # pretraining entry point
 ├── nn/                   # transformer, RoPE, flash attention
 ├── data/                 # monolingual batches, sampled subtokenizers, bit-exact resume
 ├── training/             # optimizer/schedules, checkpointing
 ├── inference/            # generation, latency benchmark
 └── evaluation/           # mcq, translation, summarization

scripts/
 ├── tokenizers/           # sbatch launchers: tokenizer training + NSL
 └── lm/                   # sbatch launchers: LM training, evaluations, latency
```

---

## Licenses

The present code is provided under the MIT license.

Portions of `modular_tokenizers/training/bpe/continued_training.py` and
`modular_tokenizers/training/utils.py` are adapted from
[tokenizer-extension](https://github.com/taidopurason/tokenizer-extension)
(Taido Purason et al., *Teaching Old Tokenizers New Words*) and remain under
the Apache License 2.0 (a copy is provided in `LICENSE-APACHE`); the
corresponding attribution is kept in those files' headers.
