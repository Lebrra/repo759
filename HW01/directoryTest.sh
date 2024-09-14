#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -J DirectoryTest
#SBATCH -o DirectoryTest.out -e DirectoryTest.err
#SBATCH -t 0-00:01:00
#SBATCH -c 2

date
echo "Directory before moving:"
pwd
cd $SLURM_SUBMIT_DIR
echo "Directory after moving:"
pwd