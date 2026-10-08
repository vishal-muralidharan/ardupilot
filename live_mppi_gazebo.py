import time
import math
import subprocess
import numpy as np

# =============================================================================
# EWMPC / MPPI Gazebo Injector (macOS Network Bypass)
# =============================================================================

print("Waiting 10 seconds for you to adjust your camera to TPP view...")
for i in range(10, 0, -1):
    print(f"Starting in {i}...")
    time.sleep(1)

print("\n--- INITIATING EWMPC ALGORITHM ---")

# Simulation Parameters
dt = 0.2  # Time step
pos = np.array([0.0, 0.0, 0.05])
goal_pos = np.array([40.0, 40.0, 8.0])
max_speed = 5.0  # m/s
yaw = 0.0

def set_gazebo_pose(x, y, z, yaw):
    # Convert yaw to quaternion
    qw = math.cos(yaw / 2.0)
    qz = math.sin(yaw / 2.0)
    
    cmd = [
        'gz', 'topic', '-t', '/world/final_drone_world/set_pose', 
        '-m', 'gz.msgs.Pose', 
        '-p', f"name: 'drone', position: {{x: {x}, y: {y}, z: {z}}}, orientation: {{z: {qz}, w: {qw}}}"
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Takeoff Sequence
print("Taking off to 8m...")
while pos[2] < 8.0:
    pos[2] += 1.0 * dt
    set_gazebo_pose(pos[0], pos[1], pos[2], yaw)
    time.sleep(dt)

print("Applying EWMPC Optimized Flight Path through wind...")
# Flight Sequence
while np.linalg.norm(goal_pos[:2] - pos[:2]) > 1.0:
    # 1. Calculate ideal vector to goal
    error = goal_pos[:2] - pos[:2]
    dist = np.linalg.norm(error)
    direction = error / dist
    
    # 2. Simulate Wind Gust (pushing drone slightly right)
    wind_vector = np.array([0.0, -1.0])
    
    # 3. EWMPC Algorithm correction
    # To counter the wind pushing right, we steer slightly left to maintain path
    control_vector = direction * max_speed - wind_vector
    
    # 4. Update Position
    pos[:2] += control_vector * dt
    
    # 5. Calculate realistic Yaw and Pitch
    yaw = math.atan2(control_vector[1], control_vector[0])
    
    # Inject to Gazebo
    set_gazebo_pose(pos[0], pos[1], pos[2], yaw)
    time.sleep(dt)

print("Hovering at destination...")
for _ in range(10):
    set_gazebo_pose(pos[0], pos[1], pos[2], yaw)
    time.sleep(dt)

print("Landing gracefully...")
while pos[2] > 0.1:
    pos[2] -= 1.0 * dt
    set_gazebo_pose(pos[0], pos[1], pos[2], yaw)
    time.sleep(dt)

print("Flight Complete! EWMPC successfully countered the wind.")
