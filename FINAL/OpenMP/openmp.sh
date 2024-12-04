#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -J FINAL_Task2
#SBATCH -o FINAL_2.out -e FINAL_2.err
#SBATCH -t 0-00:10:00
#SBATCH -c 20

module load nvidia/cuda/11.8.0
module load gcc/11.3.0
g++ rasterize.cpp filehandler.cpp pixel.cpp -Wall -O3 -std=c++17 -o final2 -fopenmp
date

./final2 $1