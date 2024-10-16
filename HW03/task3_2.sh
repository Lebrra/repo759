#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -J HW03_Task3
#SBATCH -o Task3_3.out -e Task3_3.err
#SBATCH -t 0-00:10:00
#SBATCH -c 20

g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp
date

ten=1
two=1
for ((i=0; i<6; i++)) do ten=$(($ten*10)); done
for ((i=0; i<6; i++)) do two=$(($two*2)); done

for ((t=1; t<=20; t++)) 
    do 
        ./task3 $ten $t $two; 
        echo "------------------------";
    done