#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -J FINAL_Task2
#SBATCH -o FINAL_2.out -e FINAL_2.err
#SBATCH -t 0-00:10:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1

module load nvidia/cuda/11.8.0
module load gcc/11.3.0
nvcc rasterize.cpp filehandler.cpp pixel.cpp -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o final2
date

./final2 $1