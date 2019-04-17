#!/bin/bash

total_procs=`cat /proc/cpuinfo | grep processor | wc -l`
# Taken from https://www.howtogeek.com/howto/ubuntu/display-number-of-processors-on-linux/

for num_procs in `seq ${total_procs}`
do
    echo ${num_procs}
    mpirun -n ${num_procs} python test_parallel_regression.py
done
