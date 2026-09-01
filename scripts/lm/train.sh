#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --job-name=modular_lm
#SBATCH --output=slurm_logs/%A/train.out

# Pretraining with sampled subtokenizer vocabularies (paper §4): monolingual
# batches, loss restricted to the sampled subtokenizer through a logits mask.
# One task per node owns all its GPUs (train.py claims them explicitly).
# RESUME=true continues RUN_DIR's training (bit-exact for the data stream),
# from RESUME_STEP (-1 = latest) — keep TOTAL_STEPS identical for a cosine run.
# ---- edit me (or override via environment / sbatch --export) -------------
RESUME=${RESUME:-false}
RESUME_STEP=${RESUME_STEP:--1}
LANGS=(${LANGS[@]:-en fr de})
PATHS=(${PATHS[@]:-/path/to/en.txt /path/to/fr.txt /path/to/de.txt})
WEIGHTS=(${WEIGHTS[@]:-0.4 0.3 0.3})
TOKENIZER_DIR=${TOKENIZER_DIR:-outputs/tokenizers/sequential_bpe_24000}
RUN_DIR=${RUN_DIR:-outputs/models/run_0}
DIM=${DIM:-1024}
MLP_DIM=${MLP_DIM:-2816}
N_LAYERS=${N_LAYERS:-24}
BATCH_SIZE=${BATCH_SIZE:-8}          # per process
CONTEXT_SIZE=${CONTEXT_SIZE:-2048}
TOTAL_STEPS=${TOTAL_STEPS:-50000}
LR=${LR:-5e-4}
SEED=${SEED:-1234}
P_LANG=${P_LANG:-0.5}
N_EXTRA_LANGS=${N_EXTRA_LANGS:-2}
N_SAMPLED_SUBTOKENIZERS=${N_SAMPLED_SUBTOKENIZERS:-10}
EXTRACTION=${EXTRACTION:-merged_seq_bpe}   # merged_seq_bpe (BPE) | merged_norm (unigram)
# ---------------------------------------------------------------------------

mkdir -p slurm_logs
LANGS_CSV=$(IFS=,; echo "${LANGS[*]}")

SOURCES=""
for i in "${!LANGS[@]}"; do
  SOURCES+=" data.sources.${LANGS[$i]}=${PATHS[$i]}"
  SOURCES+=" data.weights.${LANGS[$i]}=${WEIGHTS[$i]}"
done

srun uv run python -m modular_lm.train \
  run_dir=$RUN_DIR \
  resume=$RESUME \
  resume_step=$RESUME_STEP \
  distributed=true \
  seed=$SEED \
  $SOURCES \
  data.batch_size=$BATCH_SIZE \
  data.context_size=$CONTEXT_SIZE \
  data.seed=$SEED \
  model.dim=$DIM \
  model.mlp_dim=$MLP_DIM \
  model.n_layers=$N_LAYERS \
  optim.total_steps=$TOTAL_STEPS \
  optim.lr=$LR \
  tokenizer.type=modular \
  tokenizer.path=$TOKENIZER_DIR \
  tokenizer.langs=$LANGS_CSV \
  "tokenizer.sampling_strategy={strategy: uniform, p_lang: $P_LANG, n_extra_langs: $N_EXTRA_LANGS, n_sampled_subtokenizers: $N_SAMPLED_SUBTOKENIZERS}" \
  "tokenizer.extraction_strategy={strategy: $EXTRACTION}" \
  "$@"
