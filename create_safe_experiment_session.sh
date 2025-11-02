#!/bin/bash
# Check if a name argument was given
if [ -z "$1" ]; then
    echo "Usage: $0 <name>"
    exit 1
fi

NAME="$1"
SNAPSHOT="/home/stud/kgutjahr/${NAME}"

# Remove old snapshot folder if it exists
rm -rf "$SNAPSHOT"

# Copy project to snapshot folder
cp -r /home/stud/kgutjahr/STiL-TTA "$SNAPSHOT"

# Start screen session inside snapshot folder
screen -S $NAME -d -m bash -c "cd $SNAPSHOT && ./run_dist_shifts.sh; exec bash"

echo "Started screen session '$NAME' running from snapshot $SNAPSHOT"