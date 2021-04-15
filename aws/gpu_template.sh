#!/bin/bash
aws ec2 run-instances \
    --count 1 \
    --image-id ami-09f77b37a0d32243a \
    --instance-type g4dn.xlarge \
    --key-name admin-mac \
    --iam-instance-profile Name=1785-spring2021-project-trainer \
    --security-group-ids sg-06aef1369984e53f5 \
    --instance-initiated-shutdown-behavior terminate \
    --instance-market-options MarketType=spot,SpotOptions={MaxPrice=0.158}\
    --user-data "file://${INIT_SCRIPT}" \
    --output json |\
    jq -r '.Instances[].InstanceId'
