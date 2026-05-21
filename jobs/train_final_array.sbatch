#!/bin/bash
#SBATCH --job-name=sonar-final
#SBATCH --output=experiments/slurm/final_%A_%a.out
#SBATCH --error=experiments/slurm/final_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --array=0-11%4

set -e

module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.1.1
module load OpenCV/4.8.1-foss-2023a-CUDA-12.1.1-contrib

export PATH="$HOME/.local/bin:$PATH"
export DARKNET_BIN="${DARKNET_BIN:-external/yolov4/darknet/darknet}"
export YOLOV4_PRETRAINED="${YOLOV4_PRETRAINED:-external/yolov4/yolov4.conv.137}"

mkdir -p experiments/slurm

pipenv run python scripts/train_final.py \
  --job-index "${SLURM_ARRAY_TASK_ID}" \
  --darknet-bin "${DARKNET_BIN}" \
  --pretrained "${YOLOV4_PRETRAINED}"
