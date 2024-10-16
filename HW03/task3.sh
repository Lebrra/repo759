#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -J HW03_Task3
#SBATCH -o Task3_3.out -e Task3_3.err
#SBATCH -t 0-00:05:00
#SBATCH -c 20

g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp
date


./task3 6 8 2; 