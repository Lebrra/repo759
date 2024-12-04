#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -J FINAL_Task1
#SBATCH -o FINAL_1.out -e FINAL_1.err
#SBATCH -t 0-00:10:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1

module load nvidia/cuda/11.8.0
module load gcc/11.3.0
nvcc rasterize.cu filehandler.cu pixel.cu sizeAdjuster.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o final
date

./final $1