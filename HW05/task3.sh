#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -J HW05_Task3
#SBATCH -o Task5_3.out -e Task5_3.err
#SBATCH -t 0-00:05:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1

module load nvidia/cuda/11.8.0
module load gcc/11.3.0
nvcc task3.cu vscale.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task3
date

pow=1
for ((i=0; i<9; i++)) do pow=$(($pow*2)); done
for ((i=0; i<20; i++)) 
do 
    pow=$(($pow*2)); 
    ./task3 $pow; 
done