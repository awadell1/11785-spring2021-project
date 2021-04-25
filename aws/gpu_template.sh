#!/bin/bash
aws ec2 run-instances \
    --launch-template LaunchTemplateId=lt-0d443150b6cd7f552 \
    --count 1 \
    --iam-instance-profile Name=11785-spring2021-project-trainer \
    --security-group-ids sg-06aef1369984e53f5 \
    --instance-initiated-shutdown-behavior terminate \
    --instance-market-options MarketType=spot,SpotOptions={MaxPrice=0.158}\
    --user-data "file://${INIT_SCRIPT}" \
    --output json |\
    jq -r '.Instances[].InstanceId'
