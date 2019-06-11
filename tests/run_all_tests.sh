#!/bin/bash

# Input argument must be 1 if we are on Balena, and 0 if we are not.

pytest

./run_parallel_test.sh $1
