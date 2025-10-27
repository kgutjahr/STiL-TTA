#!/bin/bash

SNAPSHOT="/home/stud/kgutjahr/STiL-TTA-snapshot"

# Remove old snapshot folder if it exists
rm -rf "$SNAPSHOT"

# Copy project to snapshot folder
cp -r /home/stud/kgutjahr/STiL-TTA "$SNAPSHOT"

# Start screen session inside snapshot folder
screen -S STIL -d -m bash -c "cd $SNAPSHOT && ./run_dist_shifts.sh; exec bash"

echo "Started screen session 'STIL' running from snapshot $SNAPSHOT"