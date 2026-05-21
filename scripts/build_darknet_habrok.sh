#!/bin/bash
set -euo pipefail

module purge
module load CUDA/12.1.1
module load OpenCV/4.8.1-foss-2023a-CUDA-12.1.1-contrib

cd external/yolov4/darknet

rm -f darknet
rm -f *.o
rm -f obj/*.o
rm -f obj/*.d

make -j4 GPU=1 CUDNN=0 CUDNN_HALF=0 OPENCV=1 \
  ARCH="-gencode arch=compute_70,code=[sm_70,compute_70] -gencode arch=compute_80,code=[sm_80,compute_80]"
