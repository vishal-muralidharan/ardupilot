# ArduPilot Gazebo SITL — FYP V4 Simulation

Welcome to the FYP V4 Simulation Monorepo. This repository contains the complete simulation stack needed to test the UAV flight controller in realistic wind turbulence over Injambakkam, Chennai.

## What Has Been Done Till Now

1. **Integrated Gazebo & ArduPilot**: Successfully compiled `ardupilot_gazebo` and `ardupilot` SITL (Software-in-the-Loop) to create a headless physics engine.
2. **Created Injambakkam World**: Built a realistic 3D map of the Injambakkam coastline (`worlds/injambakkam_fyp.sdf`), including launch pads, obstacles, and a 1 km North goal.
3. **Injected Simulated Wind**: Added a 4 m/s coastal wind with 1.5 m/s turbulence into the Gazebo physics engine.
4. **Developed V4 MAVLink Bridge**: Created `fyp_v4_mavlink_bridge.py`, a script that runs the WindEKF, Adaptive LTV MPC, and Energy-aware MPPI/Greedy replanners natively, sending `LOCAL_NED` control vectors to the SITL via PyMAVLink.
5. **Implemented Energy Tracking**: Added dynamic `UAVPlantDynamics.compute_power()` to track live Instantaneous Power (W), Cumulative Energy (J), and Specific Energy Consumption (SEC in Wh/km).
6. **Built Live Dashboard**: Developed `sim_dashboard.py` to visualise path, cross-track error, latencies, and SEC live during the flight.
7. **Cleaned Workspace**: Packaged the entire environment as a single cohesive monorepo, ignoring large build files to keep the repository lightweight for teammates.

---

## How to Run the Simulations

You will need two terminal windows to run the simulation and monitor it simultaneously.

### 1. Launch the Simulation (Terminal 1)
This command boots up the Gazebo world, launches the ArduPilot SITL, and connects the V4 Python controller script.

```bash
# Default mission: MPPI replanner, 30 m altitude, 1 km north goal
./launch_simulation.sh

# Or, run with the Greedy replanner, 40 m altitude, and 800 m goal
./launch_simulation.sh greedy 40 800
```
*(Note: Gazebo runs in headless mode (`-s`). Wait a few seconds for the EKF to align and the drone to take off.)*

### 2. View the Live Dashboard (Terminal 2)
Open a new terminal in this same folder to see the real-time telemetry plots.
```bash
python3 sim_dashboard.py
```
This dashboard will show you the exact flight path overlaid on the 2D map, wind disturbance estimation, latencies, and the SEC energy graph.

---

## What Is Yet to Be Done (Next Steps)

1. **Step 1: Execute the Baseline (V3) Benchmark**
   - Run the original `baseline_v3.py` script to generate baseline benchmark data for the drone under theoretical physics.
2. **Step 2: Execute the Simulation (V4) Flight**
   - Launch `./launch_simulation.sh` and allow the drone to complete the full 1 km coastal mission, generating the `fyp_v4_telemetry.csv`.
3. **Step 3: Process the Comparison Analytics**
   - Extract `fyp_v4_telemetry.csv` and overlay it against the V3 Baseline.
   - Produce side-by-side graphical comparisons for:
     - Path variations (how the MPC handles turbulence vs the baseline).
     - Cross-track error averages.
     - Specific Energy Consumption (Wh/km) differences.
     - Computational Replanner Latency.
4. **Step 4: Final Reporting**
   - Add the final visual graphs to this documentation for the project review.
