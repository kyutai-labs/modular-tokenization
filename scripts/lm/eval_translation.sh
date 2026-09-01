#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=eval_mt
#SBATCH --output=slurm_logs/%A/eval_mt.out

# Few-shot translation evaluation (BLEU) of a trained run (paper §5).
# DATA_PATH holds {train,valid,test}.jsonl of {"question": <English>,
# "answer": [<target>]} pairs (see scripts/lm/prepare_eval_data.py):
# REVERSE=false evaluates en->target, REVERSE=true target->en.
# ---- edit me (or override via environment / sbatch --export) -------------
RUN_DIR=${RUN_DIR:-outputs/models/run_0}
STEP=${STEP:--1}
DATA_PATH=${DATA_PATH:-eval_data/flores/fr}
TARGET_LANG=${TARGET_LANG:-fr}       # BLEU tokenizer choice (zh / ja / other)
REVERSE=${REVERSE:-false}
SUBTOKENIZER_IDS=${SUBTOKENIZER_IDS:-all}
SUBTOKENIZER_ID_OUT=${SUBTOKENIZER_ID_OUT:-}
LENGTH=${LENGTH:-64}
K_SHOTS=${K_SHOTS:-5}
MAX_EXAMPLES=${MAX_EXAMPLES:--1}
OUTPUT=${OUTPUT:-outputs/evals/translation.jsonl}
# ---------------------------------------------------------------------------

mkdir -p slurm_logs "$(dirname $OUTPUT)"

uv run python -m modular_lm.evaluation.translation \
  run_dir=$RUN_DIR \
  step=$STEP \
  data_path=$DATA_PATH \
  target_lang=$TARGET_LANG \
  reverse=$REVERSE \
  subtokenizer_ids=$SUBTOKENIZER_IDS \
  ${SUBTOKENIZER_ID_OUT:+subtokenizer_id_out=$SUBTOKENIZER_ID_OUT} \
  length=$LENGTH \
  k_shots=$K_SHOTS \
  max_examples=$MAX_EXAMPLES \
  flash_attention=false \
  output=$OUTPUT
