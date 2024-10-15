#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -J HW03_Task2
#SBATCH -o Task3_2.out -e Task3_2.err
#SBATCH -t 0-00:01:00
#SBATCH -c 20

g++ task2.cpp convolution.cpp -Wall -O3 -std=c++17 -o task2 -fopenmp
date

if (($1 > 1 && $2 > 0 && $2 <= 20));
then ./task1 $1 $2;
else echo "Invalid n or t value!"; 
fi