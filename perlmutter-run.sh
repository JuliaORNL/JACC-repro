#!/bin/bash
#SBATCH -A ntrain1
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -G 1
#SBATCH -t 0:02:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

#export SLURM_CPU_BIND="cores"
PROJ_DIR=$SCRATCH/JACC-repro
PROJ_EXE=$PROJ_DIR/7-point-stencil/laplacian

module purge

module load PrgEnv-gnu
module load cray-hdf5-parallel
module load cudatoolkit/12.2

srun -n 1 --gpus=1 ./laplacian.sh $PROJ_EXE 512 8 8 8 laplacian-perlmutter-out.txt