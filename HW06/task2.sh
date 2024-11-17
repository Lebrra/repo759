#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -J HW06_Task2
#SBATCH -o Task6_2.out -e Task6_2.err
#SBATCH -t 0-00:02:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1

module load nvidia/cuda/11.8.0
module load gcc/11.3.0
nvcc task2.cu stencil.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task2
date

pow=1
for ((i=0; i<9; i++)) do pow=$(($pow*2)); done
echo "1024 threads:"
./task2 128 32 256; 
