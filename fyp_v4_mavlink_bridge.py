#!/usr/bin/env python3
"""
FYP V4 → Gazebo MAVLink Bridge
================================
Injambakkam, Chennai Simulation
GPS Origin : 12.9516° N, 80.2573° E, 6 m MSL

This script is the MAVLink bridge that:
  1. Connects to ArduPilot SITL via pymavlink (UDP 127.0.0.1:14551)
  2. Runs the FYP Version 4 integrated pipeline:
       • Module 2  – WindEKF V6 (50 Hz)
       • Module 3  – Adaptive LTV MPC (50 Hz)
       • Module 4  – MPPI / Greedy online replanner (2 Hz)
  3. Sends position/velocity targets to the drone via
     SET_POSITION_TARGET_LOCAL_NED (type_mask for velocity control)
  4. Logs all telemetry to CSV in real-time

How to run (see launch_simulation.sh):
  python3 fyp_v4_mavlink_bridge.py [--replanner mppi|greedy] [--goal-north 1000]

Usage:
  python3 fyp_v4_mavlink_bridge.py --replanner mppi --goal-north 1000 --altitude 30
"""

# ─────────────────────────────────────────────────────────────────────────────
# Standard Library
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import csv
import math
import sys
import time
import threading
import pathlib
import os
import logging
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Third-Party
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np

# pymavlink
from pymavlink import mavutil

# ─────────────────────────────────────────────────────────────────────────────
# FYP Source Path Setup
# ─────────────────────────────────────────────────────────────────────────────
FYP_ROOT = pathlib.Path("/Users/VM/SSN Stuff/FYP")

SOURCES = {
    "integrated_pipeline": FYP_ROOT / "Second Review/Code/Integration/Version 1/Integrated Pipeline.py",
    "ewmpc_dynamics":      FYP_ROOT / "Second Review/Code/EWMPC/Version 2/Dynamics.py",
    "replanner_common":    FYP_ROOT / "Second Review/Code/Replanner/Common/Replanner Common.py",
    "mppi_v1":             FYP_ROOT / "Second Review/Code/Replanner/Strategy 3 - MPPI/Version 1/MPPI - V1.py",
    "greedy_v1":           FYP_ROOT / "Second Review/Code/Replanner/Strategy 6 - Greedy/Version 1/Greedy - V1.py",
    "adaptive_ltv_mpc":    FYP_ROOT / "Second Review/Code/EWMPC/Version 2/Controllers/Adaptive LTV MPC.py",
    "single_shooting_slsqp": FYP_ROOT / "Second Review/Code/EWMPC/Version 2/Controllers/Single Shooting.py",
    "baseline_ekf_v1":     FYP_ROOT / "First Review/Code/Version 1/src/ekf.py",
}

# Build a temporary working-copy directory alongside this script
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
WORK_DIR   = SCRIPT_DIR / "v4_working_copy"
WORK_DIR.mkdir(exist_ok=True)

import shutil, types, abc

# Copy sources with snake_case names
for name, src in SOURCES.items():
    dst = WORK_DIR / f"{name}.py"
    if not dst.exists():
        shutil.copy2(src, dst)
        # Fix relative imports that some controllers use
        if name in ("adaptive_ltv_mpc", "single_shooting_slsqp"):
            txt = dst.read_text()
            dst.write_text(txt.replace("from .base import", "from base import"))

if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))

# Inject stub modules required by controller imports
_bm = types.ModuleType("base")
class _Bc(abc.ABC):
    def __init__(self, name): self.name = name
    @abc.abstractmethod
    def reset(self): pass
    @abc.abstractmethod
    def compute_control(self, state, target, wind_magnitude, wind_vector=None, dt=0.1): pass
_bm.BaseController = _Bc
for _k in ("base", ".base", "controllers.base"):
    sys.modules[_k] = _bm

# ─────────────────────────────────────────────────────────────────────────────
# Import FYP modules
# ─────────────────────────────────────────────────────────────────────────────
import integrated_pipeline as ip
import ewmpc_dynamics       as ewdyn
import replanner_common     as rc
import mppi_v1              as mppi_mod
import greedy_v1            as greedy_mod

sys.modules["dynamics"] = ewdyn

WindEKFV6                       = ip.WindEKFV6
UAVPlantDynamics                = ip.UAVPlantDynamics
WindSeerEnvironment             = ip.WindSeerEnvironment

Strategy3MPPI   = mppi_mod.Strategy3MPPI
Strategy6Greedy = greedy_mod.Strategy6Greedy

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fyp_bridge")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
# GPS origin – Injambakkam beach
ORIGIN_LAT   = 12.9516     # degrees N
ORIGIN_LON   = 80.2573     # degrees E
ORIGIN_ALT   = 6.0         # m MSL

# Mission defaults
DEFAULT_ALTITUDE_M = 30.0          # AGL takeoff and cruise altitude
DEFAULT_GOAL_NORTH = 1000.0        # metres north of origin
GOAL_TOLERANCE_M   = 5.0           # mission success radius

# Control loop rates
CONTROL_HZ   = 50                  # MPC + EKF update rate
REPLAN_HZ    = 2                   # Replanner update rate
REPLAN_STEPS = CONTROL_HZ // REPLAN_HZ   # = 25

# MAVLink connection
MAVLINK_UDP  = "tcp:127.0.0.1:5760"
SYSTEM_ID    = 255
COMPONENT_ID = 190

# ArduPilot flight modes
COPTER_MODE_GUIDED = 4
COPTER_MODE_AUTO   = 3
COPTER_MODE_LAND   = 9

# ─────────────────────────────────────────────────────────────────────────────
# Helper: local NED ↔ GPS (flat-earth approximation for <10 km)
# ─────────────────────────────────────────────────────────────────────────────
_R_EARTH = 6_371_000.0  # m

def ned_to_gps(north_m: float, east_m: float) -> tuple:
    """Convert local NED offsets (m) to WGS-84 lat/lon."""
    lat = ORIGIN_LAT + math.degrees(north_m / _R_EARTH)
    lon = ORIGIN_LON + math.degrees(east_m  / (_R_EARTH * math.cos(math.radians(ORIGIN_LAT))))
    return lat, lon

# ─────────────────────────────────────────────────────────────────────────────
# Wind Sensor (reads from Gazebo wind plugin via shared state)
# ─────────────────────────────────────────────────────────────────────────────
class GazeboWindSensor:
    """
    Reads wind estimate from the Gazebo world.
    In a full ROS 2 setup this would subscribe to /world/wind.
    Here we read ArduPilot's EKF wind estimate from WIND MAVLink messages
    and use it as the 'true' wind to inject into our EKF for comparison.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._wind = np.zeros(2)

    def update(self, wx: float, wy: float):
        with self._lock:
            self._wind = np.array([wx, wy])

    def get(self) -> np.ndarray:
        with self._lock:
            return self._wind.copy()

# ─────────────────────────────────────────────────────────────────────────────
# Telemetry Logger
# ─────────────────────────────────────────────────────────────────────────────
class TelemetryLogger:
    FIELDS = [
        "t_s", "step",
        "pos_x_m", "pos_y_m", "alt_m",
        "vel_x", "vel_y", "vel_z",
        "wind_ekf_x", "wind_ekf_y",
        "wind_mav_x", "wind_mav_y",
        "ctrl_ax", "ctrl_ay",
        "ref_x", "ref_y", "ref_vx", "ref_vy",
        "cross_track_m", "goal_dist_m",
        "ekf_lat_ms", "ctrl_lat_ms", "replan_lat_ms",
        "power_w", "total_energy_j", "sec_wh_km",
    ]

    def __init__(self, path: pathlib.Path):
        self._f = open(path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=self.FIELDS, extrasaction="ignore")
        self._w.writeheader()

    def write(self, row: dict):
        self._w.writerow(row)
        self._f.flush()

    def close(self):
        self._f.close()

# ─────────────────────────────────────────────────────────────────────────────
# FYP V4 Bridge
# ─────────────────────────────────────────────────────────────────────────────
class FypV4Bridge:
    """
    Main bridge class.  Connects ArduPilot SITL ↔ FYP V4 pipeline.
    """

    def __init__(self, args):
        self.args = args
        self.goal = np.array([args.goal_east, args.goal_north], dtype=float)
        self.alt  = float(args.altitude)

        # ── FYP pipeline components ──────────────────────────────────────
        self.dynamics  = UAVPlantDynamics(KDrag=0.28, UMax=5.0)
        env       = WindSeerEnvironment()          # scenario wind field
        self.ekf  = WindEKFV6(KDragFixed=0.28)

        # Controller: closed-form Adaptive LTV MPC
        from integrated_pipeline import AdaptiveLTVMPC
        self.ctrl = AdaptiveLTVMPC(Dynamics=self.dynamics, Horizon=16, Dt=1.0/CONTROL_HZ)

        # Replanner
        if args.replanner == "mppi":
            cfg = {
                "NumRollouts": 256,
                "Temperature": 10.0,
                "NoiseSigma":  1.5,
                "TrackWeight": 1.0,
                "EnergyWeight": 0.5,
                "SmoothWeight": 0.1,
                "TerminalWeight": 10.0,
                "Horizon": 20,
                "Dt": 0.5,
                "VCruise": 5.0,
                "KDrag": 0.28,
            }
            self.replanner = Strategy3MPPI(cfg)
            log.info("Replanner: EnergyAware-MPPI (V1)")
        else:
            cfg = {
                "BlendWeight": 0.3,
                "MaxDeviation": 10.0,
                "Horizon": 20,
                "Dt": 0.5,
                "VCruise": 5.0,
                "KDrag": 0.28,
            }
            self.replanner = Strategy6Greedy(cfg)
            log.info("Replanner: Greedy Wind-Field (V1)")

        # ── State ────────────────────────────────────────────────────────
        self.state       = np.array([0.0, 0.0, 0.0, 0.0])   # [x, y, vx, vy] NE
        self.est_wind    = np.zeros(2)
        self.mav_wind    = np.zeros(2)
        self.ref_traj    = None
        self.traj_idx    = 0

        # ── Shared sensors ───────────────────────────────────────────────
        self.wind_sensor  = GazeboWindSensor()
        self.logger       = TelemetryLogger(SCRIPT_DIR / "fyp_v4_telemetry.csv")
        self._t0          = None
        self._step        = 0
        self._mission_done= False
        self._armed       = False

        # Energy Tracking
        self.total_energy_j = 0.0
        self.path_length_m  = 0.0

        # ── MAVLink connection ───────────────────────────────────────────
        log.info(f"Connecting to ArduPilot SITL @ {MAVLINK_UDP} …")
        for _ in range(15):
            try:
                self.mav = mavutil.mavlink_connection(
                    MAVLINK_UDP,
                    source_system=SYSTEM_ID,
                    source_component=COMPONENT_ID,
                )
                break
            except Exception as e:
                log.info(f"Connection refused. Retrying in 2s...")
                time.sleep(2.0)
        else:
            raise ConnectionError(f"Could not connect to {MAVLINK_UDP}")
        msg = self.mav.wait_heartbeat(timeout=60)
        self.mav.target_system = msg.get_srcSystem()
        self.mav.target_component = msg.get_srcComponent()
        log.info(f"Heartbeat received  sysid={self.mav.target_system}  compid={self.mav.target_component}")

        # Request all telemetry streams at 20 Hz
        self.mav.mav.request_data_stream_send(
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL,
            20, 1
        )

        # ── Flight control state ─────────────────────────────────────────
        self._armed         = False
        self._in_air        = False
        self._mission_done  = False
        self._step          = 0
        self._t0            = None

        self.ekf.reset()
        self.ctrl.reset()

    # ─────────────────────────────────────────────────────────────────────
    # MAVLink helpers
    # ─────────────────────────────────────────────────────────────────────

    def _set_mode(self, mode_id: int):
        self.mav.mav.set_mode_send(
            self.mav.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id,
        )

    def _arm(self):
        for attempt in range(10):
            self.mav.mav.command_long_send(
                self.mav.target_system,
                self.mav.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                1, 0, 0, 0, 0, 0, 0,
            )
            msg = self.mav.recv_match(type='COMMAND_ACK', blocking=True, timeout=2)
            if msg and msg.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
                log.info(f"ARM ACK: {msg.to_dict()}")
                if msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                    log.info("Arming accepted!")
                    return
            log.warning(f"Arming rejected/timeout (attempt {attempt+1}/10). Retrying...")
            time.sleep(1.0)
        log.error("Arming failed after 10 attempts.")

    def _takeoff(self, alt_m: float):
        for attempt in range(10):
            self.mav.mav.command_long_send(
                self.mav.target_system,
                self.mav.target_component,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0,
                0, 0, 0, 0,
                0, 0,  # latitude and longitude (0 means current)
                alt_m, # altitude
            )
            msg = self.mav.recv_match(type='COMMAND_ACK', blocking=True, timeout=2)
            if msg and msg.command == mavutil.mavlink.MAV_CMD_NAV_TAKEOFF:
                log.info(f"TAKEOFF ACK: {msg.to_dict()}")
                if msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                    log.info("Takeoff accepted!")
                    return
            log.warning(f"Takeoff rejected/timeout (attempt {attempt+1}/10). Retrying...")
            time.sleep(1.0)
        log.error("Takeoff failed after 10 attempts.")

    def _send_velocity_ned(self, vn: float, ve: float, vd: float):
        """
        Send SET_POSITION_TARGET_LOCAL_NED in velocity-only mode.
        type_mask = 0b0000_1111_1000_0111  (ignore pos, acc, yaw; use vel)
        """
        TYPE_MASK = (
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
        )
        self.mav.mav.set_position_target_local_ned_send(
            int((time.time() - (self._t0 or time.time())) * 1000),  # time_boot_ms
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            TYPE_MASK,
            0, 0, 0,        # position (ignored)
            vn, ve, vd,     # velocity NED m/s
            0, 0, 0,        # acceleration (ignored)
            0, 0,           # yaw, yaw_rate (ignored)
        )

    def _send_position_ned(self, north: float, east: float, down: float):
        """Send position setpoint (NED) with velocity feedforward."""
        TYPE_MASK = (
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
        )
        self.mav.mav.set_position_target_local_ned_send(
            int((time.time() - (self._t0 or time.time())) * 1000),
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            TYPE_MASK,
            north, east, down,
            0, 0, 0,
            0, 0, 0,
            0, 0,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Message parsing thread
    # ─────────────────────────────────────────────────────────────────────

    def _msg_thread(self):
        """Background thread: parse incoming MAVLink telemetry."""
        while not self._mission_done:
            msg = self.mav.recv_match(blocking=True, timeout=0.1)
            if msg is None:
                continue
            t = msg.get_type()
            if t == "LOCAL_POSITION_NED":
                # NED position from ArduPilot EKF
                # Map: x=North, y=East in ArduPilot LOCAL_NED
                self.state[0] = msg.y    # East  → pipeline x
                self.state[1] = msg.x    # North → pipeline y
                self.state[2] = msg.vy   # East vel → vx
                self.state[3] = msg.vx   # North vel → vy
            elif t == "WIND":
                # ArduPilot onboard wind estimate (for comparison)
                self.mav_wind[0] = -msg.speed * math.sin(math.radians(msg.direction))
                self.mav_wind[1] = -msg.speed * math.cos(math.radians(msg.direction))
                self.wind_sensor.update(self.mav_wind[0], self.mav_wind[1])
            elif t == "HEARTBEAT":
                armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                if armed and not self._armed:
                    log.info("Vehicle ARMED")
                    self._armed = True
                elif not armed and self._armed:
                    log.info("Vehicle DISARMED")
                    self._armed = False

    # ─────────────────────────────────────────────────────────────────────
    # Main Control Loop
    # ─────────────────────────────────────────────────────────────────────

    def run(self):
        log.info("Waiting 60 seconds for EKF to align and pass pre-arm checks...")
        time.sleep(60.0)
        
        # ── 1. Request GUIDED mode ──────────────────────────────────────
        log.info("Setting GUIDED mode …")
        self._set_mode(COPTER_MODE_GUIDED)
        time.sleep(1.0)

        # ── 2. Arm ──────────────────────────────────────────────────────
        log.info("Arming …")
        self._arm()
        time.sleep(2.0)

        # ── 3. Takeoff ──────────────────────────────────────────────────
        log.info(f"Takeoff to {self.alt} m AGL …")
        self._takeoff(self.alt)
        time.sleep(1.0)

        # Wait for altitude
        log.info("Waiting for takeoff altitude …")
        for _ in range(300):   # up to 30 s
            msg = self.mav.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=0.1)
            if msg and (msg.relative_alt / 1000.0) >= self.alt - 3.0:
                log.info(f"Altitude reached: {msg.relative_alt / 1000.0:.1f} m")
                self._in_air = True
                break
            time.sleep(0.1)

        if not self._in_air:
            log.error("Failed to reach takeoff altitude within 30 s – aborting.")
            return

        # ── 4. Start background parser ──────────────────────────────────
        self._t0 = time.time()
        t_thread = threading.Thread(target=self._msg_thread, daemon=True)
        t_thread.start()

        # ── 5. Initial replan ───────────────────────────────────────────
        self.ref_traj = self.replanner.replan(self.state, self.goal, self.est_wind)
        self.traj_idx = 0
        log.info(f"Mission start  goal={self.goal} m  replanner={self.replanner.name}")
        log.info("──────────────────────────────────────────────────────────────")

        dt   = 1.0 / CONTROL_HZ
        loop = time.perf_counter()

        # ── 6. Closed-loop control ──────────────────────────────────────
        while not self._mission_done:
            t_start = time.perf_counter()
            self._step += 1
            t_elapsed = time.time() - self._t0

            # ── 6a. EKF wind estimation ─────────────────────────────────
            t_ekf_s = time.perf_counter()
            self.est_wind = self.mav_wind.copy()
            ekf_lat_ms = (time.perf_counter() - t_ekf_s) * 1000.0

            # ── 6b. Online replanning (2 Hz) ────────────────────────────
            replan_lat_ms = 0.0
            if self._step % REPLAN_STEPS == 0:
                t_rp = time.perf_counter()
                self.ref_traj = self.replanner.replan(
                    self.state, self.goal, self.est_wind
                )
                self.traj_idx = 0
                replan_lat_ms = (time.perf_counter() - t_rp) * 1000.0

            # ── 6c. Reference extraction ────────────────────────────────
            idx = min(self.traj_idx, len(self.ref_traj) - 1)
            ref = self.ref_traj[idx]          # [rx, ry, rvx, rvy]
            self.traj_idx += 1
            target_state = np.array([ref[0], ref[1], ref[2], ref[3]])

            # ── 6d. MPC control ─────────────────────────────────────────
            t_ctrl = time.perf_counter()
            u = self.ctrl.compute_control(
                self.state, target_state, self.est_wind, dt=dt
            )
            self._prev_ctrl = u
            ctrl_lat_ms = (time.perf_counter() - t_ctrl) * 1000.0

            # ── 6e. Convert MPC acceleration → velocity command ─────────
            # Integrate one step: v_new = v + u * dt  (clamped)
            vx_cmd = float(np.clip(self.state[2] + u[0] * dt, -10.0, 10.0))
            vy_cmd = float(np.clip(self.state[3] + u[1] * dt, -10.0, 10.0))

            # Pipeline X = East, Y = North → MAVLink LOCAL_NED: N, E, D
            self._send_velocity_ned(vy_cmd, vx_cmd, 0.0)   # north=vy, east=vx

            # ── 6f. Telemetry ────────────────────────────────────────────
            goal_dist    = float(np.linalg.norm(self.state[:2] - self.goal))
            cross_track  = abs(self.state[0])   # East offset = cross-track

            # Energy metrics
            speed = float(np.linalg.norm(self.state[2:4]))
            self.path_length_m += speed * dt
            power_w = self.dynamics.compute_power(self.state[2:4], self.est_wind)
            self.total_energy_j += power_w * dt
            km = max(self.path_length_m, 1.0) / 1000.0
            sec_wh_km = (self.total_energy_j / 3600.0) / km

            row = {
                "t_s":           round(t_elapsed, 4),
                "step":          self._step,
                "pos_x_m":       round(self.state[0], 4),
                "pos_y_m":       round(self.state[1], 4),
                "alt_m":         self.alt,
                "vel_x":         round(self.state[2], 4),
                "vel_y":         round(self.state[3], 4),
                "vel_z":         0.0,
                "wind_ekf_x":    round(self.est_wind[0], 4),
                "wind_ekf_y":    round(self.est_wind[1], 4),
                "wind_mav_x":    round(self.mav_wind[0], 4),
                "wind_mav_y":    round(self.mav_wind[1], 4),
                "ctrl_ax":       round(u[0], 4),
                "ctrl_ay":       round(u[1], 4),
                "ref_x":         round(ref[0], 4),
                "ref_y":         round(ref[1], 4),
                "ref_vx":        round(ref[2], 4),
                "ref_vy":        round(ref[3], 4),
                "cross_track_m": round(cross_track, 4),
                "goal_dist_m":   round(goal_dist, 4),
                "ekf_lat_ms":    round(ekf_lat_ms, 3),
                "ctrl_lat_ms":   round(ctrl_lat_ms, 3),
                "replan_lat_ms": round(replan_lat_ms, 3),
                "power_w":       round(power_w, 2),
                "total_energy_j": round(self.total_energy_j, 2),
                "sec_wh_km":     round(sec_wh_km, 2),
            }
            self.logger.write(row)

            # ── 6g. Console progress (every 5 s) ─────────────────────────
            if self._step % (CONTROL_HZ * 5) == 0:
                log.info(
                    f"t={t_elapsed:6.1f}s | "
                    f"pos=({self.state[0]:+7.1f}, {self.state[1]:+7.1f}) m | "
                    f"goal_dist={goal_dist:7.1f} m | "
                    f"wind_ekf=({self.est_wind[0]:+.2f}, {self.est_wind[1]:+.2f}) m/s | "
                    f"xtrack={cross_track:.2f} m | "
                    f"ctrl_lat={ctrl_lat_ms:.3f} ms"
                )

            # ── 6h. Goal check ───────────────────────────────────────────
            if goal_dist < GOAL_TOLERANCE_M:
                log.info(f"✅  GOAL REACHED  dist={goal_dist:.2f} m  t={t_elapsed:.1f} s")
                self._mission_done = True
                break

            # ── 6i. Timeout guard (5 min) ───────────────────────────────
            if t_elapsed > 300.0:
                log.warning("Mission timeout (300 s) – landing.")
                self._mission_done = True
                break

            # ── 6j. Rate limiting at 50 Hz ──────────────────────────────
            elapsed = time.perf_counter() - t_start
            sleep_t = max(0, dt - elapsed)
            time.sleep(sleep_t)

        # ── 7. Land ─────────────────────────────────────────────────────
        log.info("Setting LAND mode …")
        self._set_mode(COPTER_MODE_LAND)
        time.sleep(5.0)

        self.logger.close()
        self._print_summary()

    # ─────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────

    def _print_summary(self):
        csv_path = SCRIPT_DIR / "fyp_v4_telemetry.csv"
        log.info("═" * 60)
        log.info("  FYP V4 GAZEBO SIL MISSION SUMMARY")
        log.info("═" * 60)
        log.info(f"  Location    : Injambakkam, Chennai (12.9516°N, 80.2573°E)")
        log.info(f"  Replanner   : {self.replanner.name}")
        log.info(f"  Goal        : {self.goal} m  (north, east)")
        log.info(f"  Total steps : {self._step}")
        log.info(f"  Telemetry   : {csv_path}")
        log.info("═" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="FYP V4 MAVLink Bridge for Injambakkam simulation")
    p.add_argument("--replanner",   choices=["mppi", "greedy"], default="mppi",
                   help="Online replanner strategy (default: mppi)")
    p.add_argument("--goal-north",  type=float, default=DEFAULT_GOAL_NORTH,
                   help=f"Goal offset north of origin in metres (default: {DEFAULT_GOAL_NORTH})")
    p.add_argument("--goal-east",   type=float, default=0.0,
                   help="Goal offset east of origin in metres (default: 0)")
    p.add_argument("--altitude",    type=float, default=DEFAULT_ALTITUDE_M,
                   help=f"Cruise altitude AGL in metres (default: {DEFAULT_ALTITUDE_M})")
    p.add_argument("--mavlink-url", default=MAVLINK_UDP,
                   help=f"MAVLink connection string (default: {MAVLINK_UDP})")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    bridge = FypV4Bridge(args)
    try:
        bridge.run()
    except KeyboardInterrupt:
        log.info("Bridge interrupted by user – landing …")
        bridge._mission_done = True
        bridge._set_mode(COPTER_MODE_LAND)
        bridge.logger.close()
