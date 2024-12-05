#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -J HW07_Task2
#SBATCH -o Task7_2.out -e Task7_2.err
#SBATCH -t 0-00:05:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1

module load nvidia/cuda/11.8.0
module load gcc/11.3.0
nvcc task2.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task2
date

./task1 512 512 1;