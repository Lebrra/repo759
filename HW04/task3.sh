#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -J HW04_Task3
#SBATCH -o Task4_3.out -e Task4_3.err
#SBATCH -t 0-00:01:00
#SBATCH -c 8

g++ task3.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp
date

for ((t=1; t<=8; t++)) 
do
./task3 100 10 $t
done