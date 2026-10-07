import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load V4 telemetry
df = pd.read_csv("fyp_v4_telemetry.csv")

# Create a stunning 2D overhead trajectory map
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

# Plot reference line
ax.plot([0, 0], [0, 1000], color='#444444', linestyle='--', linewidth=2, label='Optimal Path (0m Cross-Track)')

# Plot V4 Trajectory
ax.plot(df['pos_x_m'], df['pos_y_m'], color='#00ffcc', linewidth=3, label='FYP V4 (Energy-Aware MPPI)')

# Decorate
ax.set_title("1km Sprint Trajectory - Gazebo SITL", fontsize=18, fontweight='bold', color='white', pad=20)
ax.set_xlabel("East (m)", fontsize=14)
ax.set_ylabel("North (m)", fontsize=14)
ax.grid(True, color='#222222', linestyle=':')
ax.legend(loc='upper left', fontsize=12, frameon=False)
ax.set_aspect('equal')

# Save
plt.tight_layout()
plt.savefig("v4_trajectory_map.png", dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
print("Plot saved to v4_trajectory_map.png")
