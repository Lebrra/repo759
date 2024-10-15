#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -J HW03_Task2
#SBATCH -o Task3_2.out -e Task3_2.err
#SBATCH -t 0-00:05:00
#SBATCH -c 20

g++ task2.cpp convolution.cpp -Wall -O3 -std=c++17 -o task2 -fopenmp
date

if (($1 > 1));
then for ((t=1; t<=20; t++)) do ./task2 $1 $t; done
else echo "Invalid n value!"; 
fi