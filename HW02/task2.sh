#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -J HW02_Task2
#SBATCH -o Task2_2.out -e Task2_2.err
#SBATCH -t 0-00:05:00
#SBATCH --gres=gpu:1 -c 1

module load gcc/13.2.0
module load nvidia/cuda
g++ convolution.cpp task2.cpp -Wall -O3 -std=c++17 -o task2

date

if (($1 > 0 && $2 > 0 && $2 % 2));
then ./task2 $1 $2; 
else echo "Invalid n or m value!"; 
fi