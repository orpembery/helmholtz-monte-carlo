#!/bin/bash

# Input argument must be 1 if we are on Balena, and 0 if we are not.

# Following idea nabbed from https://intoli.com/blog/exit-on-errors-in-bash-scripts/
# Learned about exit codes from https://bencane.com/2014/09/02/understanding-exit-codes-and-how-to-use-them-in-bash-scripts/
set -e

total_procs=`cat /proc/cpuinfo | grep processor | wc -l`
# Taken from https://www.howtogeek.com/howto/ubuntu/display-number-of-processors-on-linux/

for num_procs in `seq ${total_procs}`
do
    echo ${num_procs}
    mpirun -n ${num_procs} python test_parallel_regression.py $1

    mpirun -n ${num_procs} python test_parallel_coeff_gather.py $1

    mpirun -n ${num_procs} python test_parallel_point_evaluation.py $1
done
