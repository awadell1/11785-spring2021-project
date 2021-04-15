#!/bin/bash
echo "Uptime: $(cat /proc/uptime)"

# CONFIG
USER=ubuntu

# Activate conda environment
echo 'conda activate pytorch_latest_p37' >> /home/$USER/.bashrc

# Ensure files are private
echo 'umask 077' >> /home/$USER/.bashrc
echo 'umask 077' >> /home/$USER/.profile

echo "Adding cron job for to push runs"
echo "*/15 * * * * cd ~/$REPO && make runs-push" | crontab -u "$USER" -

# Write run script to disk
cat <<EOF > run_script.sh
${SCRIPT}
EOF
mv run_script.sh /home/$USER/run_script.sh
chown --reference /home/$USER/.bashrc /home/$USER/run_script.sh
chmod u+x /home/$USER/run_script.sh
su - $USER -c /home/$USER/run_script.sh &

echo "marking init as done"
echo "Uptime: $(cat /proc/uptime)"
sudo -u ubuntu touch /home/ubuntu/instance-init-done
