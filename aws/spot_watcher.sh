#!/bin/bash
# spot_watcher.sh cmd
# Will run cmd, while polling AWS for spot termination
#
# If cmd returns -> Return
# If AWS initiates shutdown -> Send SIGTERM to cmd, and then return

# Start cmd and log pid
eval $@ &
pid=$!

# Function to check spot -> Exit with 0 iff no termination
function check_spot(){
    status=$(curl -qI http://169.254.169.254/latest/meta-data/spot/termination-time --max-time 5 2>/dev/null | head -n1)
    if [ "$status" == *"200"* ]; then
        return 1
    else
        return 0
    fi
}

# Monitor spot and process
while check_spot && kill -0 $pid ; do
    : # busy-wait
done
echo "Killing $pid"
kill $pid
exit 0
