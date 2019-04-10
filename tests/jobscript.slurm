#!/bin/bash

#SBATCH --job-name=test
#SBATCH --account=free
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --time=00:05:00
#SBATCH --error=%j.err
#SBATCH --output=%j.out
#SBATCH --partition=batch-all

source /beegfs/scratch/user/s/orp20/own-utilities/helmholtz_firedrake_venv.sh

mpirun -n 32 python /beegfs/scratch/user/s/orp20/helmholtz-monte-carlo/tests/test_parallel_regression.py