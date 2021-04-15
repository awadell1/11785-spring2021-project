#!/bin/bash

function create_runner(){
COMMAND=$1
instance="aws/gpu_template.sh"
if [ $# -gt 1 ]; then
    run_dir="runner/$2"
    mkdir -p $run_dir
else
    mkdir -p runner
    run_dir=$(mktemp -d runner/XXXXXXXX | xargs)
fi

# Get Git Info
export BRANCH=$(git branch --show-current)
export REPO=$(basename "$(realpath .)")

# Check that we've pushed
HEAD=$(git rev-parse $BRANCH)
REMOTE=$(git rev-parse origin/$BRANCH | tail -n1)
if [ "$HEAD" != "$REMOTE" ]; then
    echo "$BRANCH has not been pushed to origin -> exiting"
    exit 1
fi

# Build init script
export COMMAND
export INIT_SCRIPT=$(realpath $run_dir/init-script.sh)
export SCRIPT=$(cat aws/run_template.sh | envsubst '${REPO} ${COMMAND} ${BRANCH}')
cat aws/init_template.sh | envsubst '${SCRIPT} ${REPO}' > "$INIT_SCRIPT"

# Build add worker script
cat aws/gpu_template.sh | envsubst '${INIT_SCRIPT}' > $run_dir/add_worker.sh
chmod u+x ./$run_dir/*.sh

echo "Created runner scripts in $run_dir"
}
