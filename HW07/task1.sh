#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -J HW07_Task1
#SBATCH -o Task7_1.out -e Task7_1.err
#SBATCH -t 0-00:05:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1

module load nvidia/cuda/11.8.0
module load gcc/11.3.0
nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -stdc++17 -o task1
date

pow=1
for ((i=0; i<4; i++)) do pow=$(($pow*2)); done
for ((i=0; i<10; i++)) 
do 
    pow=$(($pow*2)); 
    ./task1 $pow 512;
done