#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=0
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --job-name=mono_unigram
#SBATCH --output=slurm_logs/%A/mono_unigram_%a.out
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

# Monolingual Unigram tokenizers (paper §3.1, step 1).
# BACKEND=hf (default, writes .json) or sentencepiece (paper-exact, writes .model).
BACKEND=${BACKEND:-hf}
lang=${LANGS[$SLURM_ARRAY_TASK_ID]}
input=${PATHS[$SLURM_ARRAY_TASK_ID]}

uv run python -m modular_tokenizers.training.unigram.train \
  inputs=$input model_prefix=$OUT/monolingual_unigram_$BACKEND/$lang \
  vocab_size=$VOCAB_SIZE backend=$BACKEND
