#!/bin/bash
#SBATCH --job-name=yolov4-iou
#SBATCH --output=experiments/slurm/yolov4_iou_%A_%a.out
#SBATCH --error=experiments/slurm/yolov4_iou_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=0-5%3

set -e

module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.1.1
module load OpenCV/4.8.1-foss-2023a-CUDA-12.1.1-contrib

export PATH="$HOME/.local/bin:$PATH"

mkdir -p experiments/slurm

pipenv run python scripts/evaluate_yolov4_iou_sweep.py \
  --job-index "${SLURM_ARRAY_TASK_ID}" \
  --darknet-bin external/yolov4/darknet/darknet
