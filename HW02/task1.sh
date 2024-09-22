#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -J HW02_Task1
#SBATCH -o Task2_1.out -e Task2_1.err
#SBATCH -t 0-00:05:00
#SBATCH --gres=gpu:1 -c 1

module load gcc/13.2.0
module load nvidia/cuda
g++ scan.cpp task1.cpp -Wall -O3 -std=c++17 -o task1

date

pow=1
for ((i=0; i<9; i++)) do pow=$(($pow*2)); done
for ((i=0; i<20; i++)) 
do 
    pow=$(($pow*2)); 
    ./task1 $pow; 
done