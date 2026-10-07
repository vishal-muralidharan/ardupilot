# FYP V4 Simulation — Future Steps and Roadmap

This document outlines the detailed plan to complete the comparative evaluation between the **Baseline (V1)** and the **Adaptive LTV MPC (V4)** UAV flight pipelines under the realistic coastal turbulence of Injambakkam, Chennai.

---

## Phase 1: Baseline Generation (V1 Benchmark)

**Objective:** Establish a reference dataset for flight path deviation and energy consumption using the legacy V1 control logic under mathematical turbulence.

### 1.1 Environment Verification
- Locate the original `baseline_v3.py` inside the Integration workspace.
- Ensure the mathematical environment parameters match the Gazebo conditions exactly (e.g., 4 m/s base wind with a 1.5 m/s turbulent gust variation).

### 1.2 Execution
- Run `python3 baseline_v3.py` for a 1 km North trajectory mission.
- Monitor the theoretical aerodynamic drag inversions and ensure the legacy PID/LQR controller attempts to correct the path.

### 1.3 Data Logging
- Confirm the script successfully dumps its output into `baseline_v3_telemetry.csv`.
- Essential captured metrics must include:
  - Timestamp (`t_s`)
  - Coordinates (`pos_x_m`, `pos_y_m`)
  - Aerodynamic Energy (`total_energy_kj`)
  - Specific Energy Consumption (`sec_wh_km`)

---

## Phase 2: Simulation Generation (V4 Flight)

**Objective:** Run the full FYP V4 pipeline inside the Gazebo Harmonic + ArduPilot SITL physical physics engine to generate the experimental dataset.

### 2.1 SITL Calibration
- Launch the simulation environment using `./launch_simulation.sh`.
- Wait for the ArduPilot SITL's EKF (Extended Kalman Filter) to align and acquire a valid GPS lock at the Injambakkam launch pad.

### 2.2 MPPI Flight Execution
- Observe the MAVLink bridge arming the vehicle and executing a takeoff to the target altitude.
- Allow the V4 pipeline (running `WindEKFV6`, `AdaptiveLTVMPC`, and the `Strategy3MPPI` replanner) to autonomously complete the 1 km delivery mission.
- Use `sim_dashboard.py` to visually confirm that the drone successfully counters the onshore wind disturbances.

### 2.3 Greedy Flight Execution (Optional/Comparative)
- Re-run the simulation using the Greedy replanner (`./launch_simulation.sh greedy 30 1000`).
- Ensure telemetry is logged under a distinct filename for later comparison.

### 2.4 Data Validation
- Verify the generation of `fyp_v4_telemetry.csv`.
- Inspect the file to ensure the physical power computation (via `UAVPlantDynamics.compute_power`) and the MAVLink `LOCAL_NED` coordinates were logged synchronously at 50 Hz.

---

## Phase 3: Analytics and Data Processing

**Objective:** Write a comparative script to cross-analyze the Baseline vs V4 datasets and extract meaningful metrics to prove the efficacy of the V4 upgrade.

### 3.1 Data Alignment
- Create an analysis script (e.g., `analytics_comparator.py`) using `pandas` and `matplotlib`.
- Normalize the timestamps of both datasets, interpolating values if the baseline script ran at a different loop rate than the 50 Hz Gazebo MAVLink bridge.

### 3.2 Performance Metrics Extraction
- **Cross-Track Error (RMSE):** Compute the Root Mean Square Error of the vehicle's deviation from the pure North 1 km line for both V1 and V4.
- **Energy Savings Percentage:** Calculate the difference in final Total Energy (J) and SEC (Wh/km) between V1 and V4. A positive delta proves the energy-aware MPPI successfully generated a more efficient path.
- **Latency Overheads:** Calculate the average computing time (in ms) of the V4 Adaptive MPC and the MPPI Replanner over the lifespan of the flight.

### 3.3 Visual Plotting
- Generate high-DPI (e.g., 300 DPI) side-by-side graphical overlays:
  1. **Trajectory Map:** 2D East-North map showing both flight paths overlapping, visually proving which pipeline held a tighter line.
  2. **Error Graph:** Cross-track error (m) plotted over Time (s).
  3. **Energy Graph:** SEC (Wh/km) plotted over Time (s).

---

## Phase 4: Final Reporting

**Objective:** Consolidate the findings into the final project documentation and thesis.

- Embed the high-DPI analytical plots directly into the repository documentation.
- Formulate a concluding argument demonstrating how the integration of real-time Wind EKF estimation and closed-form Adaptive LTV MPC resulted in statistically significant reductions in cross-track error and aerodynamic energy drag compared to the legacy PID/LQR baseline.
