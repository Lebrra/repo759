#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -J HW06_Task1
#SBATCH -o Task6_1.out -e Task6_1.err
#SBATCH -t 0-00:05:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1

module load nvidia/cuda/11.8.0
module load gcc/11.3.0
nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task1
date

./task1 4 1024