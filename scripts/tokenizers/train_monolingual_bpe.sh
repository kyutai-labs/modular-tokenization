#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=0
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --job-name=mono_bpe
#SBATCH --output=slurm_logs/%A/mono_bpe_%a.out
#SBATCH --array=0-20
# ---- edit me -----------------------------------------------------------
# One entry per language, aligned index-wise. Any text source works: plain
# .txt, .zst, JSONL (add text_field=... below), or a directory of such files.
LANGS=(bg cs da de el en es fr hr hu it ne nl pl pt ro si sl so sw te)
PATHS=(/path/to/bg.txt /path/to/cs.txt ...)
OUT=outputs/tokenizers
VOCAB_SIZE=24000
# -------------------------------------------------------------------------

mkdir -p slurm_logs

# Monolingual BPE references (one per language; the NSL denominators of Tab. 1).
lang=${LANGS[$SLURM_ARRAY_TASK_ID]}
input=${PATHS[$SLURM_ARRAY_TASK_ID]}

uv run python -m modular_tokenizers.training.bpe.train_base \
  input_path=$input vocab_size=$VOCAB_SIZE \
  tokenizer_json_path=$OUT/monolingual_bpe/$lang.json
