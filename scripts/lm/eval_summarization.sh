#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=eval_sum
#SBATCH --output=slurm_logs/%A/eval_sum.out

# Few-shot summarization evaluation (ROUGE) of a trained run (paper §5).
# DATA_PATH holds train.jsonl + test.jsonl in the dataset's original format;
# TASK selects the field mapping (xlsum | eur_lex_sum | wiki_lingua).
# ---- edit me (or override via environment / sbatch --export) -------------
RUN_DIR=${RUN_DIR:-outputs/models/run_0}
STEP=${STEP:--1}
TASK=${TASK:-xlsum}
DATA_PATH=${DATA_PATH:-eval_data/xlsum/en}
SUBTOKENIZER_IDS=${SUBTOKENIZER_IDS:-en}
LENGTH=${LENGTH:-64}
SOURCE_MAX_LENGTH=${SOURCE_MAX_LENGTH:-512}
K_SHOTS=${K_SHOTS:-5}
MAX_EXAMPLES=${MAX_EXAMPLES:--1}
OUTPUT=${OUTPUT:-outputs/evals/summarization.jsonl}
# ---------------------------------------------------------------------------

mkdir -p slurm_logs "$(dirname $OUTPUT)"

uv run python -m modular_lm.evaluation.summarization \
  run_dir=$RUN_DIR \
  step=$STEP \
  task=$TASK \
  data_path=$DATA_PATH \
  subtokenizer_ids=$SUBTOKENIZER_IDS \
  length=$LENGTH \
  source_max_length=$SOURCE_MAX_LENGTH \
  k_shots=$K_SHOTS \
  max_examples=$MAX_EXAMPLES \
  flash_attention=false \
  output=$OUTPUT
