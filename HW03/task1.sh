#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -J HW03_Task1
#SBATCH -o Task3_1.out -e Task3_1.err
#SBATCH -t 0-00:01:00
#SBATCH -c 20

g++ task1.cpp matmul.cpp -Wall -O3 -std=c++17 -o task1 -fopenmp
date

if not $2;
then for ((t=1; t<=20; t++)) do ./task1 $1 $t; done
elif (($1 > 1 && $2 > 0 && $2 <= 20));
then ./task1 $1 $2;
else echo "Invalid n or t value!"; 
fi