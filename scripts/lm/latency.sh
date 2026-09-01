#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=latency
#SBATCH --output=slurm_logs/%A/latency.out

# Forward-pass latency vs vocabulary size (paper §5, efficiency): the same
# architecture measured with a subtokenizer's compact vocabulary and, with
# USE_GLOBAL_VOCAB=true, with the full merged vocabulary on the same tokens.
# ---- edit me (or override via environment / sbatch --export) -------------
DATA_PATH=${DATA_PATH:-/path/to/text.txt}
TOKENIZER_DIR=${TOKENIZER_DIR:-outputs/tokenizers/sequential_bpe_24000}
LANGS=${LANGS:-en}
EXTRACTION=${EXTRACTION:-merged_seq_bpe}
SUBTOKENIZER_ID=${SUBTOKENIZER_ID:-en}
USE_GLOBAL_VOCAB=${USE_GLOBAL_VOCAB:-false}
CONTEXT_SIZES=${CONTEXT_SIZES:-1024,2048,4096}
DIM=${DIM:-1024}
MLP_DIM=${MLP_DIM:-2816}
N_LAYERS=${N_LAYERS:-24}
OUTPUT=${OUTPUT:-outputs/evals/latency.jsonl}
# ---------------------------------------------------------------------------

mkdir -p slurm_logs "$(dirname $OUTPUT)"

uv run python -m modular_lm.inference.latency \
  data_path=$DATA_PATH \
  context_sizes=$CONTEXT_SIZES \
  jit=true \
  subtokenizer_id=$SUBTOKENIZER_ID \
  use_global_vocab=$USE_GLOBAL_VOCAB \
  model.dim=$DIM \
  model.mlp_dim=$MLP_DIM \
  model.n_layers=$N_LAYERS \
  tokenizer.type=modular \
  tokenizer.path=$TOKENIZER_DIR \
  tokenizer.langs=$LANGS \
  "tokenizer.extraction_strategy={strategy: $EXTRACTION}" \
  output=$OUTPUT
