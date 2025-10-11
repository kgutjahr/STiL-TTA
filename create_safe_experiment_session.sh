#!/bin/bash

SNAPSHOT="/home/kgutjahr/STiL-TTA-snapshot"

# Remove old snapshot folder if it exists
rm -rf "$SNAPSHOT"

# Copy project to snapshot folder
cp -r /home/kgutjahr/STiL-TTA "$SNAPSHOT"

# Start screen session inside snapshot folder
screen -S STiL_all_labelled -d -m bash -c "cd $SNAPSHOT && ./run_dist_shifts.sh; exec bash"

echo "Started screen session 'STiL_all_labelled' running from snapshot $SNAPSHOT"