#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -J HW01_Task6
#SBATCH -o Task6.out -e Task6.err
#SBATCH -t 0-00:01:00
#SBATCH --gres=gpu:1 -c 1

cd $SLURM_SUBMIT_DIR
module load gcc/13.2.0
module load nvidia/cuda

date
echo $1

g++ task6.cpp -Wall -O3 -std=c++17 -o task6
./task6 $1