#!/bin/bash

#SBATCH --job-name=test
#SBATCH --account=rc-ma1200
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --time=00:45:00
#SBATCH --error=%j.err
#SBATCH --output=%j.out
#SBATCH --partition=batch-all

source /beegfs/scratch/user/s/orp20/own-utilities/helmholtz_firedrake_venv.sh

/beegfs/scratch/user/s/orp20/helmholtz-monte-carlo/tests/run_parallel_test.sh 1