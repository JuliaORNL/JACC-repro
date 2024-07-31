#!/bin/bash
#SBATCH -A TRN025
#SBATCH -J cpu_test
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 0:10:00
#SBATCH -p batch
#SBATCH -N 1

date

module load amd
module load cray-mpich

PROJ_DIR='/ccsopen/home/y1e/cpu'
L=$1

cd $PROJ_DIR

srun -n 1 -c 64 --threads-per-core=2 ./test.sh $L