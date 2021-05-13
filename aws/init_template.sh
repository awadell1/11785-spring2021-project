#!/bin/bash
echo "Uptime: $(cat /proc/uptime)"

# CONFIG
USER=ubuntu

# Ensure files are private
echo 'umask 077' >> /home/$USER/.bashrc
echo 'umask 077' >> /home/$USER/.profile

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
