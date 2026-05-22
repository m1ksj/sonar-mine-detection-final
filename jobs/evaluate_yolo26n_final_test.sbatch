#!/bin/bash
#SBATCH --job-name=yolo26n-test
#SBATCH --output=experiments/slurm/yolo26n_test_%A_%a.out
#SBATCH --error=experiments/slurm/yolo26n_test_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=6-11%3

set -e

module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.1.1

export PATH="$HOME/.local/bin:$PATH"

mkdir -p experiments/slurm

pipenv run python scripts/evaluate_yolo26n_final_test.py \
  --job-index "${SLURM_ARRAY_TASK_ID}"
