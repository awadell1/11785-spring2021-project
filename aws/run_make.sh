#!/bin/bash
# run_make.sh MAKE_TARGET
# Gennerates run script for running a make recipe on EC2

# Confirm recipe exists
if ! output=$(make -n "$@") ; then
    echo "$output"
    exit 1
fi

# Build Run Command
read -r -d '' COMMAND << EOF
# Run Make Recipe
nohup make -j $@ > train.log &
EOF

# Create Runner
source aws/create_runner.sh
create_runner "$COMMAND"
