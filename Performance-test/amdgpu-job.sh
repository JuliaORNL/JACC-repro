#!/bin/bash
#SBATCH -A CSC594
#SBATCH -J AMDPerformanceTest
#SBATCH -o %x-%j.out
#SBATCH -t 00:05:00
#SBATCH -p batch
#SBATCH -N 1

# project directory with the executable
cd $PROJWORK/csc594/ygtang/amd-lab-notes/finite-difference/examples

# input size
L=512
# block dimension
Bx=128
By=8
Bz=1
for run in {1..100}; do
        srun -n1 --gpus=1 ./laplacian_dp_kernel1 $L $L $L $Bx $By $Bz
done