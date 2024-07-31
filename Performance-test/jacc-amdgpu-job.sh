#!/bin/bash
#SBATCH -A csc594
#SBATCH -J gs-julia-1MPI-1GPU
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 0:10:00
#SBATCH -N 1

date

GS_DIR=/lustre/orion/proj-shared/csc594/ygtang/GrayScott.jl
GS_EXE=$GS_DIR/gray-scott.jl
GS_IN=$GS_DIR/examples/settings-files.json

srun -n 1 --gpus=1 julia --project=$GS_DIR $GS_EXE $GS_IN