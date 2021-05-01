#!/bin/bash
# Train model on EC2
# ./aws/train.sh MAKE_TRAIN_TARGET ARGS...

DIR=$(dirname $(realpath $0))
target=$1
shift

# Escape optional arguments
aws/run_make.sh $target ARGS="'$*'"
