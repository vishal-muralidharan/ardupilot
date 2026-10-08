import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load Telemetry
v4_df = pd.read_csv("fyp_v4_mppi_telemetry.csv")
baseline_df = pd.read_csv("fyp_v4_telemetry.csv")

# Extract Baseline Data (Greedy)
baseline_cross_track = baseline_df["cross_track_m"].values
baseline_rmse = np.sqrt(np.mean(baseline_cross_track**2))
baseline_energy = baseline_df["total_energy_j"].iloc[-1]
baseline_flight_time = baseline_df["t_s"].iloc[-1]

# Extract V4 Data (MPPI)
v4_cross_track = v4_df["cross_track_m"].values
v4_rmse = np.sqrt(np.mean(v4_cross_track**2))
v4_energy = v4_df["total_energy_j"].iloc[-1]
v4_flight_time = v4_df["t_s"].iloc[-1]

# Calculate Improvements
rmse_improvement = ((baseline_rmse - v4_rmse) / baseline_rmse) * 100
energy_improvement = ((baseline_energy - v4_energy) / baseline_energy) * 100

print("="*60)
print(f"       FYP Analytics: V1 Greedy vs V4 MPPI (Gazebo SITL)")
print("="*60)
print(f"{'Metric':<25} | {'V1 Greedy':<12} | {'V4 MPPI':<12} | {'Improvement'}")
print("-" * 60)
print(f"{'Cross-track RMSE (m)':<25} | {baseline_rmse:<12.2f} | {v4_rmse:<12.2f} | {rmse_improvement:+.1f}%")
print(f"{'Total Energy (Joules)':<25} | {baseline_energy:<12.1f} | {v4_energy:<12.1f} | {energy_improvement:+.1f}%")
print(f"{'Flight Time (s)':<25} | {baseline_flight_time:<12.1f} | {v4_flight_time:<12.1f} | {((baseline_flight_time - v4_flight_time)/baseline_flight_time)*100:+.1f}%")
print("="*60)

# Generate Side-by-Side RMSE Plot
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

labels = ['V1 Greedy Baseline', 'V4 Energy-Aware MPPI']
values = [baseline_rmse, v4_rmse]
colors = ['#ff4444', '#00ccff']

bars = ax.bar(labels, values, color=colors, width=0.5)
ax.set_ylabel('Cross-track RMSE (meters)', fontsize=12)
ax.set_title('Cross-track Error in Gazebo Turbulence', fontsize=14, fontweight='bold', color='white', pad=20)

for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f} m',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig("rmse_comparison.png", dpi=300, facecolor=fig.get_facecolor())
print("\nPlot saved to rmse_comparison.png")
