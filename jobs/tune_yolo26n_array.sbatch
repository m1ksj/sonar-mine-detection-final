#!/bin/bash
#SBATCH --job-name=yolo26n-tune
#SBATCH --output=experiments/slurm/yolo26n_tune_%A_%a.out
#SBATCH --error=experiments/slurm/yolo26n_tune_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --array=0-59%10

set -e

module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.1.1

export PATH="$HOME/.local/bin:$PATH"

mkdir -p experiments/slurm

pipenv run python scripts/tune_yolo26n.py --job-index "${SLURM_ARRAY_TASK_ID}"
