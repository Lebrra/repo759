#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -J HW03_Task1
#SBATCH -o Task3_1.out -e Task3_1.err
#SBATCH -t 0-00:05:00
#SBATCH --gres=gpu:1 -c 1
#SBATCH --cpus-per-task=20

module load gcc/13.2.0
module load nvidia/cuda
g++ task1.cpp matmul.cpp -Wall -O3 -std=c++17 -o task1 -fopenmp
date

if (($1 > 1 && $2 > 0 && $2 <= 20));
then ./task1 $1 $2;
else echo "Invalid n or t value!"; 
fi