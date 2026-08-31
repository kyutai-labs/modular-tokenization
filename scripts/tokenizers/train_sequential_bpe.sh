#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=0
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --job-name=seq_bpe
#SBATCH --output=slurm_logs/%A/seq_bpe_%a.out
#SBATCH --array=0-0
# ---- edit me -----------------------------------------------------------
# One entry per language, aligned index-wise. Any text source works: plain
# .txt, .zst, JSONL (add text_field=... below), or a directory of such files.
LANGS=(bg cs da de el en es fr hr hu it ne nl pl pt ro si sl so sw te)
PATHS=(/path/to/bg.txt /path/to/cs.txt ...)
OUT=outputs/tokenizers
VOCAB_SIZE=24000
# -------------------------------------------------------------------------

mkdir -p slurm_logs

# Sequential BPE (paper §3.2, Algorithm 1): one multilingual tokenizer built
# language by language (alphabetical order in the paper), with a per-language
# vocabulary budget. Writes the tokenizer directory (tokenizer.json,
# lang_to_ids.json, extracted/<lang>/).
LANGS_CSV=$(IFS=,; echo "${LANGS[*]}")
PATHS_CSV=$(IFS=,; echo "${PATHS[*]}")

uv run python -m modular_tokenizers.training.bpe.sequential \
  input_paths=$PATHS_CSV langs_order=$LANGS_CSV \
  vocab_size=$VOCAB_SIZE output_dir=$OUT/sequential_bpe_$VOCAB_SIZE
