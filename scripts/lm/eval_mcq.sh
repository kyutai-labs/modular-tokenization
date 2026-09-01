#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=eval_mcq
#SBATCH --output=slurm_logs/%A/eval_mcq.out

# Multiple-choice evaluation of a trained run, per subtokenizer (paper §5).
# TASKS and DATA_PATHS are aligned index-wise; each data path is that task's
# own directory (train.jsonl + test.jsonl — see scripts/lm/prepare_eval_data.py).
# ---- edit me (or override via environment / sbatch --export) -------------
RUN_DIR=${RUN_DIR:-outputs/models/run_0}
STEP=${STEP:--1}
TASKS=(${TASKS[@]:-arc arc csqa piqa siqa hellaswag})
DATA_PATHS=(${DATA_PATHS[@]:-eval_data/arc/easy eval_data/arc/challenge eval_data/csqa eval_data/piqa eval_data/siqa eval_data/hellaswag})
SUBTOKENIZER_IDS=${SUBTOKENIZER_IDS:-en}
MAX_EXAMPLES=${MAX_EXAMPLES:--1}
OUTPUT=${OUTPUT:-outputs/evals/mcq.jsonl}
# ---------------------------------------------------------------------------

mkdir -p slurm_logs "$(dirname $OUTPUT)"

for i in "${!TASKS[@]}"; do
  uv run python -m modular_lm.evaluation.mcq \
    run_dir=$RUN_DIR \
    step=$STEP \
    task=${TASKS[$i]} \
    data_path=${DATA_PATHS[$i]} \
    subtokenizer_ids=$SUBTOKENIZER_IDS \
    max_examples=$MAX_EXAMPLES \
    flash_attention=true \
    output=$OUTPUT
done
