#!/bin/bash
#SBATCH --job-name=sonar-final
#SBATCH --output=experiments/slurm/final_%A_%a.out
#SBATCH --error=experiments/slurm/final_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --array=0-11

set -e

mkdir -p experiments/slurm

python scripts/train_final.py \
  --job-index "${SLURM_ARRAY_TASK_ID}" \
  --darknet-bin "${DARKNET_BIN:-}" \
  --pretrained "${YOLOV4_PRETRAINED:-}"
