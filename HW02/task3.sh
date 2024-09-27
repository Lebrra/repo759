#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -J HW02_Task2
#SBATCH -o Task2_2.out -e Task2_2.err
#SBATCH -t 0-00:05:00
#SBATCH --gres=gpu:1 -c 1

module load gcc/13.2.0
module load nvidia/cuda
g++ task3.cpp matmul.cpp -Wall -O3 -std=c++17 -o task3
date
./task3