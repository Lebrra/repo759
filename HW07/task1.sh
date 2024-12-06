#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -J HW07_Task1
#SBATCH -o Task7_1.out -e Task7_1.err
#SBATCH -t 0-00:05:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1

module load nvidia/cuda/11.8.0
module load gcc/11.3.0
nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task1
date

n=512
b=512

./task1 $n $b 1;
./task1 $n $b 2;
./task1 $n $b 3;