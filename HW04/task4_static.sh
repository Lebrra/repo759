#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -J HW04_Task4
#SBATCH -o Task4_4.out -e Task4_4.err
#SBATCH -t 0-00:08:00
#SBATCH -c 8

g++ task4_static.cpp -Wall -O3 -std=c++17 -o task4 -fopenmp
date

for ((t=1; t<=8; t++)) 
do
./task4 300 100 $t
done