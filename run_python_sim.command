#!/usr/bin/env bash
cd /Users/vishalmuralidharan/Applications/Gazebo
echo "Starting Python Visual Simulation..."
/Users/vishalmuralidharan/.pyenv/versions/3.10.18/bin/python standalone_drone_sim.py
echo "Simulation finished. Press any key to close this window."
read -n 1
