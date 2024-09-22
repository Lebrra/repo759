#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -J HW02_Task1
#SBATCH -o Task2_1.out -e Task2_1.err
#SBATCH -t 0-00:01:00
#SBATCH --gres=gpu:1 -c 1

module load gcc/13.2.0
module load nvidia/cuda

date

set pow=2
for ((i=0; i<29; i++)) 
do pow=pow*2; 
done

for ((i=0; i<10; i++)) 
do 
pow=pow*2;
echo "$pow ";
done