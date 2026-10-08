#!/usr/bin/env bash
cd /Users/vishalmuralidharan/Applications/Gazebo

# Ensure Homebrew is in PATH so the 'gz' command is found
export PATH="/opt/homebrew/bin:$PATH"

# Clean up old processes
pkill -9 -f "gz sim"

# Launch Server and GUI separately (macOS requirement)
# We unset GZ_IP to avoid UDP permission issues and rely on defaults
unset GZ_IP
export GZ_PARTITION=simple_test
export GZ_SIM_RESOURCE_PATH="/Users/vishalmuralidharan/Applications/Gazebo/ardupilot_gazebo/models:/opt/homebrew/share/gz/gz-sim8/worlds:/opt/homebrew/share/gz/gz-common5/media/materials/textures"

echo "Starting Simple Gazebo Server..."
gz sim -s -r simple_flight.sdf >/dev/null 2>&1 &
SERVER_PID=$!

sleep 2

echo "Starting Gazebo GUI..."
gz sim -g

echo "Cleaning up..."
kill -9 $SERVER_PID

echo "Simulation closed. Press Enter to exit."
read

