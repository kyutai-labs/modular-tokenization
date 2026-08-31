#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=nsl
#SBATCH --output=slurm_logs/%A/nsl.out

# NSL (paper Eq. 1 / Tab. 1): the modular tokenizer's compression per language
# vs monolingual references, on your evaluation texts (e.g. FLORES devtest).
# All paths live in the YAML config -- copy nsl.example.yaml and edit it.
CONFIG=${CONFIG:-scripts/tokenizers/nsl.example.yaml}

mkdir -p slurm_logs
uv run python -m modular_tokenizers.evaluation.nsl config=$CONFIG
