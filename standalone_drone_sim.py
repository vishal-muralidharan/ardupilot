import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =============================================================================
# Standalone Ideal Drone Simulation with Dynamic Wind
# =============================================================================

# Simulation Parameters
dt = 0.1  # Time step
start_pos = np.array([0.0, 0.0])
goal_pos = np.array([1000.0, 1000.0])
max_speed = 15.0  # m/s

# Drone State
pos = np.copy(start_pos)
vel = np.array([0.0, 0.0])

# PID Controller (Simple proportional navigation)
Kp = 0.1
max_accel = 5.0

# Dynamic Wind Field
def get_wind(position, t):
    # Base wind blowing South-West
    base_wind = np.array([-4.0, -3.0])
    
    # Add dynamic sinusoidal gusts based on position and time
    gust_x = 2.0 * np.sin(position[1] / 100.0 + t)
    gust_y = 3.0 * np.cos(position[0] / 150.0 + t * 1.5)
    
    return base_wind + np.array([gust_x, gust_y])

# Data tracking for plotting
path_x, path_y = [pos[0]], [pos[1]]
time_elapsed = 0.0

# Setup Plot
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-100, 1200)
ax.set_ylim(-100, 1200)
ax.set_title("Ideal Drone Simulation (Live Physics & Dynamic Wind)", fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel("East (m)")
ax.set_ylabel("North (m)")
ax.grid(True, alpha=0.2)

# Draw Goal
ax.plot(*goal_pos, 'g*', markersize=15, label="Goal")
ax.plot(*start_pos, 'wo', markersize=8, label="Start")

# Animated Elements
drone_marker, = ax.plot([], [], 'ro', markersize=10, label="Drone")
path_line, = ax.plot([], [], 'c--', alpha=0.6, label="Trajectory")
wind_quiver = ax.quiver([], [], [], [], color='cyan', alpha=0.5, scale=50, label="Dynamic Wind")

# Text UI
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white', fontsize=12)
dist_text = ax.text(0.02, 0.91, '', transform=ax.transAxes, color='white', fontsize=12)
wind_text = ax.text(0.02, 0.87, '', transform=ax.transAxes, color='cyan', fontsize=12)

ax.legend(loc="lower right")

# Create a static grid of points for wind visualization
grid_x, grid_y = np.meshgrid(np.linspace(0, 1000, 10), np.linspace(0, 1000, 10))

def update(frame):
    global pos, vel, time_elapsed
    
    # 1. Calculate Control Command (PID to Goal)
    error = goal_pos - pos
    dist_to_goal = np.linalg.norm(error)
    
    if dist_to_goal < 5.0:
        return drone_marker, path_line, time_text, dist_text, wind_text
        
    desired_vel = (error / dist_to_goal) * max_speed
    accel_cmd = (desired_vel - vel) * Kp
    
    # Clamp acceleration
    accel_norm = np.linalg.norm(accel_cmd)
    if accel_norm > max_accel:
        accel_cmd = (accel_cmd / accel_norm) * max_accel
        
    # 2. Get Dynamic Wind at current position
    current_wind = get_wind(pos, time_elapsed)
    
    # 3. Physics Update (Euler Integration)
    # The drone tries to move with accel_cmd, but the wind pushes it!
    vel += accel_cmd * dt
    
    # Limit max velocity relative to air
    vel_norm = np.linalg.norm(vel)
    if vel_norm > max_speed:
        vel = (vel / vel_norm) * max_speed
        
    # Actual movement includes wind disturbance
    pos += (vel + current_wind) * dt
    time_elapsed += dt
    
    # Save path
    path_x.append(pos[0])
    path_y.append(pos[1])
    
    # Update visuals
    drone_marker.set_data([pos[0]], [pos[1]])
    path_line.set_data(path_x, path_y)
    
    time_text.set_text(f"Time: {time_elapsed:.1f} s")
    dist_text.set_text(f"Distance to Goal: {dist_to_goal:.1f} m")
    wind_text.set_text(f"Local Wind: {current_wind[0]:.1f}, {current_wind[1]:.1f} m/s")
    
    # Update Wind Grid Vectors
    u = np.zeros_like(grid_x)
    v = np.zeros_like(grid_y)
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            w = get_wind(np.array([grid_x[i,j], grid_y[i,j]]), time_elapsed)
            u[i,j] = w[0]
            v[i,j] = w[1]
            
    wind_quiver.set_offsets(np.column_stack([grid_x.flatten(), grid_y.flatten()]))
    wind_quiver.set_UVC(u.flatten(), v.flatten())
    
    return drone_marker, path_line, wind_quiver, time_text, dist_text, wind_text

# Run animation at 30 FPS
ani = FuncAnimation(fig, update, frames=2000, interval=33, blit=False, repeat=False)

plt.show()
