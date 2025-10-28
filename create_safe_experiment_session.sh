#!/bin/bash

SNAPSHOT="/home/stud/kgutjahr/STiL-consent-loss-snapshot"

# Remove old snapshot folder if it exists
rm -rf "$SNAPSHOT"

# Copy project to snapshot folder
cp -r /home/stud/kgutjahr/STiL-TTA "$SNAPSHOT"

# Start screen session inside snapshot folder
screen -S STIL-consent-loss -d -m bash -c "cd $SNAPSHOT && ./run_dist_shifts.sh; exec bash"

echo "Started screen session 'STIL-consent-loss' running from snapshot $SNAPSHOT"