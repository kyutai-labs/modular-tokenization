#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=0
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --job-name=merge_unigram
#SBATCH --output=slurm_logs/%A/merge_unigram_%a.out
#SBATCH --array=0-0
# ---- edit me (or override via environment / sbatch --export) -------------
# One entry per language, aligned index-wise. Supported sources: plain text,
# JSONL (add text_field=... below), .zst versions of either, or a directory.
LANGS=(${LANGS[@]:-bg cs da de el en es fr hr hu it ne nl pl pt ro si sl so sw te})
PATHS=(${PATHS[@]:-/path/to/bg.txt /path/to/cs.txt ...})
OUT=${OUT:-outputs/tokenizers}
VOCAB_SIZE=${VOCAB_SIZE:-24000}
# ---------------------------------------------------------------------------

mkdir -p slurm_logs

# Merged Unigram (paper §3.1, steps 2-3): vocabulary union of the monolingual
# tokenizers, then EM re-estimation of the piece probabilities over the
# combined corpus, applied back into the tokenizer directory.
#
# NOTE on EM cost: the EM step is pure Python and scales with
# sentences x positions x vocabulary. Run it on a SMALL sample and cap the
# iterations (settings below took ~13 min for 21 languages / ~400k pieces,
# and are sufficient to reproduce the paper's compression results);
# full-corpus EM would take weeks.
BACKEND=${BACKEND:-hf}
EXT=$([ "$BACKEND" = hf ] && echo .json || echo .model)
LANGS_CSV=$(IFS=,; echo "${LANGS[*]}")
PATHS_CSV=$(IFS=,; echo "${PATHS[*]}")
sep=""; MODELS=$(for l in "${LANGS[@]}"; do printf "%s$OUT/monolingual_unigram_$BACKEND/${l}$EXT" $sep; sep=,; done)
DIR=$OUT/merged_unigram_$BACKEND

uv run python -m modular_tokenizers.training.unigram.merge \
  model_paths=$MODELS \
  langs=$LANGS_CSV \
  output_dir=$DIR

uv run python -m modular_tokenizers.training.unigram.em_reestimate \
  data_path=$PATHS_CSV \
  tokenizers_path=$MODELS \
  save_path=$DIR/em_scores \
  max_input_sentences=10000 \
  max_iters=50 \
  tol=1e-4 \
  verbose=true \
  apply_to_model=$DIR/tokenizer$EXT \
  out_model=$DIR/tokenizer$EXT
