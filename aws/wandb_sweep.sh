#!/bin/bash
# wandb_sweep.sh CONFIG_YAML N

# Init Sweep
WB_PROJECT=$(wandb status | tail -n+2 | jq -r .project)
[ ! -z "$WB_PROJECT" ] || (echo "wandb not inited" && exit 1)
echo "Starting sweep for $WB_PROJECT"

# Sweep file
CONFIG_YAML=$1
[ -f "$CONFIG_YAML" ] || (echo "$CONFIG_YAML is not a file" && exit 1)
export SWEEP_CMD=$(wandb sweep --project "$WB_PROJECT" "$CONFIG_YAML" 2>&1 | \
    grep "Run sweep agent with" | cut -f 3 -d':' | xargs)
echo "$SWEEP_CMD"
sweep_id=$( echo "$SWEEP_CMD" | cut -d'/' -f3 )

# Build Run Command
read -r -d '' COMMAND << EOF
# Run Sweep
export SWEEP_CMD="${SWEEP_CMD}"
nohup make -j spot-train-sweep > train.log &
EOF

# Create Runner
source aws/create_runner.sh
create_runner "$COMMAND" "$sweep_id"
