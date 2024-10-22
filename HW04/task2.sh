#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -J HW04_Task2
#SBATCH -o Task4_2.out -e Task4_2.err
#SBATCH -t 0-00:01:00
#SBATCH -c 8

g++ task2.cpp -Wall -O3 -std=c++17 -o task2
date

./task2 10 10