#!/bin/bash

# Input argument must be 1 if we are on Balena, and 0 if we are not.

total_procs=`cat /proc/cpuinfo | grep processor | wc -l`
# Taken from https://www.howtogeek.com/howto/ubuntu/display-number-of-processors-on-linux/

for num_procs in `seq ${total_procs}`
do
    echo ${num_procs}
    mpirun -n ${num_procs} python test_parallel_regression.py $1

    mpirun -n ${num_procs} python test_parallel_coeff_gather.py $1

    mpirun -n ${num_procs} python test_parallel_point_evaluation.py $1
done
