#!/usr/bin/env bash
cd /Users/vishalmuralidharan/Applications/Gazebo

# Add Homebrew to PATH
export PATH="/opt/homebrew/bin:$PATH"

# Clean up
pkill -9 -f "gz sim"

# Clean network
unset GZ_IP
export GZ_PARTITION=final_flight

# Resource path for mesh
export GZ_SIM_RESOURCE_PATH="/Users/vishalmuralidharan/Applications/Gazebo/ardupilot_gazebo/models:/opt/homebrew/share/gz/gz-sim8/worlds:/opt/homebrew/share/gz/gz-common5/media/materials/textures"

echo "Starting Gazebo Server (Final Simulation)..."
gz sim -s -r final_drone_sim.sdf >/dev/null 2>&1 &
SERVER_PID=$!

sleep 2

echo "Starting Live EWMPC Python Bridge (10s delay)..."
/Users/vishalmuralidharan/.pyenv/versions/3.10.18/bin/python live_mppi_gazebo.py &
PYTHON_PID=$!

echo "Starting Gazebo GUI..."
gz sim -g

echo "Cleaning up..."
kill -9 $SERVER_PID
kill -9 $PYTHON_PID

echo "Simulation Closed. Press Enter to exit."
read
