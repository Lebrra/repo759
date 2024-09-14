#!/usr/bin/env zsh

#SBATCH -p instruction

#SBATCH -J FirstSlurm

#SBATCH -o FirstSlurm.out -e FirstSlurm.err

#SBATCH -t 0-00:01:00

#SBATCH -c 2

cd $SLURM_SUBMIT_DIR

date
hostname