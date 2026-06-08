#!/bin/bash
#SBATCH --job-name=darknet-test
#SBATCH --output=experiments/slurm/darknet_test_%j.out
#SBATCH --error=experiments/slurm/darknet_test_%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1

set -e

module purge
module load CUDA/12.1.1
module load OpenCV/4.8.1-foss-2023a-CUDA-12.1.1-contrib

mkdir -p experiments/slurm

nvidia-smi
external/yolov4/darknet/darknet 2>&1 | head -40
