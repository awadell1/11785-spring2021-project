#!/bin/bash
exec >> $0.log
exec 2>&1
echo "Uptime: \$(cat /proc/uptime)"

# Init Conda
source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch_latest_p37

# Set default region to ohio
aws configure set region us-east-2

# Configure ssh
mkdir -p ~/.ssh
eval \`ssh-agent\`
aws ssm get-parameters --with-decryption --name "$REPO-deploy" | jq -r ".Parameters[].Value" > ~/.ssh/github-deploy
ssh-add ~/.ssh/github-deploy
ssh-keyscan github.com >> ~/.ssh/known_hosts

# Clone Repo from Github
git clone "git@github.com:awadell1/$REPO.git" --branch "${BRANCH}" --depth 1

# Install python packages and login to services
python -m pip install wandb kaggle
mkdir -p ~/.kaggle
aws ssm get-parameters --with-decryption --name kaggle-api | jq -r ".Parameters[].Value" > ~/.kaggle/kaggle.json
wandb_api=\$(aws ssm get-parameters --with-decryption --name wandb-api  | jq -r ".Parameters[].Value")
wandb login "\$wandb_api"

cd ~/${REPO}
${COMMAND}

echo "done"
echo "Uptime: \$(cat /proc/uptime)"
touch ~/training-started
