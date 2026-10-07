#!/usr/bin/env python3
"""
FYP V4 Simulation Dashboard
=============================
Live-updating Matplotlib visualiser for the Injambakkam simulation.
Reads fyp_v4_telemetry.csv and plots:
  • 2-D flight path (East-North map) with wind arrows
  • Wind EKF estimate vs ArduPilot internal wind
  • Cross-track error over time
  • Control latencies (EKF, MPC, Replanner)
  • Goal distance over time

Usage:
  python3 sim_dashboard.py [--csv path/to/fyp_v4_telemetry.csv] [--refresh 1.0]
"""

import argparse
import pathlib
import time
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CSV = pathlib.Path(__file__).parent / "fyp_v4_telemetry.csv"

PALETTE = {
    "path":       "#00e5ff",
    "ref":        "#ff9800",
    "wind_ekf":   "#69f0ae",
    "wind_mav":   "#ef5350",
    "xtrack":     "#ce93d8",
    "goal":       "#fff176",
    "ekf_lat":    "#69f0ae",
    "ctrl_lat":   "#00e5ff",
    "replan_lat": "#ff9800",
    "bg":         "#0d1117",
    "fg":         "#e6edf3",
    "grid":       "#21262d",
}

# Injambakkam landmarks (East m, North m) – relative to origin
LANDMARKS = {
    "Launch Pad":    (0, 0),
    "Goal (~1 km N)": (0, 1000),
    "Water Tower":   (-150, 900),
    "ECR Road":      (100, 500),
}

# ─────────────────────────────────────────────────────────────────────────────
# Data Reader
# ─────────────────────────────────────────────────────────────────────────────

def read_telemetry(csv_path: pathlib.Path) -> dict:
    """Read CSV, return dict of arrays. Returns empty dict if file not ready."""
    data = {k: [] for k in [
        "t_s", "pos_x_m", "pos_y_m",
        "wind_ekf_x", "wind_ekf_y", "wind_mav_x", "wind_mav_y",
        "cross_track_m", "goal_dist_m",
        "ekf_lat_ms", "ctrl_lat_ms", "replan_lat_ms",
        "ref_x", "ref_y", "power_w", "sec_wh_km"
    ]}
    if not csv_path.exists():
        return data
    try:
        with open(csv_path, "r") as f:
            import csv as _csv
            reader = _csv.DictReader(f)
            for row in reader:
                try:
                    for k in data:
                        data[k].append(float(row.get(k, 0) or 0))
                except (ValueError, KeyError):
                    continue
    except Exception:
        pass
    return {k: np.array(v) for k, v in data.items()}

# ─────────────────────────────────────────────────────────────────────────────
# Dashboard
# ─────────────────────────────────────────────────────────────────────────────

def setup_figure():
    plt.style.use("dark_background")
    matplotlib.rcParams.update({
        "font.family": "monospace",
        "font.size": 9,
        "axes.facecolor": PALETTE["bg"],
        "figure.facecolor": PALETTE["bg"],
        "axes.edgecolor": PALETTE["grid"],
        "axes.labelcolor": PALETTE["fg"],
        "xtick.color": PALETTE["fg"],
        "ytick.color": PALETTE["fg"],
        "text.color": PALETTE["fg"],
        "grid.color": PALETTE["grid"],
        "axes.grid": True,
        "grid.alpha": 0.4,
    })

    fig = plt.figure(figsize=(16, 9), facecolor=PALETTE["bg"])
    fig.canvas.manager.set_window_title(
        "FYP V4 – Injambakkam Simulation Dashboard"
    )
    gs = gridspec.GridSpec(
        4, 3,
        figure=fig,
        left=0.06, right=0.97,
        top=0.93, bottom=0.07,
        wspace=0.35, hspace=0.65,
    )

    ax_map    = fig.add_subplot(gs[:, 0])    # Column 0: 2-D map (tall)
    ax_wind   = fig.add_subplot(gs[0, 1:])   # Row 0: wind EKF vs MAV
    ax_xtrack = fig.add_subplot(gs[1, 1])    # Row 1: cross-track
    ax_goal   = fig.add_subplot(gs[1, 2])    # Row 1: goal distance
    ax_lat    = fig.add_subplot(gs[2, 1:])   # Row 2: latencies
    ax_nrg    = fig.add_subplot(gs[3, 1:])   # Row 3: energy metrics

    return fig, ax_map, ax_wind, ax_xtrack, ax_goal, ax_lat, ax_nrg

# ─────────────────────────────────────────────────────────────────────────────
# Title banner
# ─────────────────────────────────────────────────────────────────────────────
TITLE = (
    "FYP V4  ·  Injambakkam, Chennai  ·  12.9516°N 80.2573°E  "
    "·  WindEKF V6 + Adaptive LTV MPC + MPPI/Greedy Replanner"
)

def update(frame, csv_path, fig, ax_map, ax_wind, ax_xtrack, ax_goal, ax_lat, ax_nrg):
    d = read_telemetry(csv_path)
    n = len(d["t_s"])

    fig.suptitle(
        TITLE + f"  ·  {n} steps",
        fontsize=9, color=PALETTE["fg"], y=0.98,
    )

    # ── Map ───────────────────────────────────────────────────────────────
    ax_map.clear()
    ax_map.set_facecolor(PALETTE["bg"])
    ax_map.set_title("Mission Map (East–North)", color=PALETTE["fg"], fontsize=9, pad=6)
    ax_map.set_xlabel("East (m)")
    ax_map.set_ylabel("North (m)")
    ax_map.set_xlim(-300, 300)
    ax_map.set_ylim(-50, 1100)
    ax_map.set_aspect("equal")

    # Buildings silhouettes
    for bx, by, bw, bh, _ in [
        (80, 150, 25, 20, 0), (-60, 220, 18, 22, 0),
        (120, 350, 30, 15, 0), (-90, 480, 20, 20, 0),
        (50, 620, 22, 18, 0), (-30, 780, 15, 25, 0),
    ]:
        ax_map.add_patch(
            mpatches.Rectangle(
                (bx - bw/2, by - bh/2), bw, bh,
                linewidth=0, facecolor="#2a2a3a", alpha=0.7,
            )
        )

    # Landmarks
    for name, (ex, ny) in LANDMARKS.items():
        c = "#69f0ae" if "Goal" in name else "#ff9800" if "Launch" in name else "#aaa"
        ax_map.plot(ex, ny, "o", color=c, markersize=8 if "Goal" in name or "Launch" in name else 5, zorder=5)
        ax_map.annotate(name, (ex, ny), textcoords="offset points",
                        xytext=(6, 4), fontsize=7, color=c)

    # Reference trajectory
    if n > 0:
        ax_map.plot(d["ref_x"], d["ref_y"], "--",
                    color=PALETTE["ref"], lw=0.8, alpha=0.5, label="Ref. Traj.")

    # Actual path
    if n > 0:
        sc = ax_map.scatter(
            d["pos_x_m"], d["pos_y_m"],
            c=np.arange(n), cmap="cool", s=2, zorder=4, label="UAV Path"
        )
        # Current position
        ax_map.plot(d["pos_x_m"][-1], d["pos_y_m"][-1],
                    "w^", markersize=10, zorder=6, markeredgewidth=1.5)

        # Wind arrows (subsample)
        if n > 10:
            idx = np.linspace(0, n - 1, min(15, n), dtype=int)
            ax_map.quiver(
                d["pos_x_m"][idx], d["pos_y_m"][idx],
                d["wind_ekf_x"][idx], d["wind_ekf_y"][idx],
                color=PALETTE["wind_ekf"], alpha=0.7,
                scale=30, width=0.006, label="Wind EKF",
            )

    ax_map.legend(loc="lower right", fontsize=7, framealpha=0.3)

    # ── Wind EKF vs MAVLink ───────────────────────────────────────────────
    ax_wind.clear()
    ax_wind.set_facecolor(PALETTE["bg"])
    ax_wind.set_title("Wind Estimation  (EKF V6  vs  ArduPilot WIND msg)", fontsize=9, pad=4)
    ax_wind.set_xlabel("Time (s)")
    ax_wind.set_ylabel("Wind (m/s)")
    if n > 1:
        ax_wind.plot(d["t_s"], d["wind_ekf_x"], color=PALETTE["wind_ekf"],
                     lw=1.2, alpha=0.9, label="EKF wₓ (East)")
        ax_wind.plot(d["t_s"], d["wind_ekf_y"], color=PALETTE["wind_ekf"],
                     lw=1.2, alpha=0.5, ls="--", label="EKF wᵧ (North)")
        ax_wind.plot(d["t_s"], d["wind_mav_x"], color=PALETTE["wind_mav"],
                     lw=0.8, alpha=0.7, label="MAV wₓ")
        ax_wind.plot(d["t_s"], d["wind_mav_y"], color=PALETTE["wind_mav"],
                     lw=0.8, alpha=0.4, ls="--", label="MAV wᵧ")
    ax_wind.axhline(0, color="#444", lw=0.5)
    ax_wind.legend(loc="upper right", fontsize=7, framealpha=0.3, ncol=2)

    # ── Cross-Track Error ─────────────────────────────────────────────────
    ax_xtrack.clear()
    ax_xtrack.set_facecolor(PALETTE["bg"])
    ax_xtrack.set_title("Cross-Track Error (m)", fontsize=9, pad=4)
    ax_xtrack.set_xlabel("Time (s)")
    ax_xtrack.set_ylabel("|Δ East| (m)")
    if n > 1:
        ax_xtrack.fill_between(d["t_s"], d["cross_track_m"],
                               color=PALETTE["xtrack"], alpha=0.4)
        ax_xtrack.plot(d["t_s"], d["cross_track_m"],
                       color=PALETTE["xtrack"], lw=1.2)
        rmse = np.sqrt(np.mean(d["cross_track_m"] ** 2))
        ax_xtrack.set_title(
            f"Cross-Track Error  RMSE={rmse:.2f} m", fontsize=9, pad=4
        )

    # ── Goal Distance ────────────────────────────────────────────────────
    ax_goal.clear()
    ax_goal.set_facecolor(PALETTE["bg"])
    ax_goal.set_title("Distance to Goal (m)", fontsize=9, pad=4)
    ax_goal.set_xlabel("Time (s)")
    ax_goal.set_ylabel("Distance (m)")
    if n > 1:
        ax_goal.plot(d["t_s"], d["goal_dist_m"],
                     color=PALETTE["goal"], lw=1.5)
        ax_goal.axhline(5.0, color="#f44", lw=0.8, ls="--", label="5 m threshold")
        ax_goal.legend(fontsize=7, framealpha=0.3)

    # ── Latencies ────────────────────────────────────────────────────────
    ax_lat.clear()
    ax_lat.set_facecolor(PALETTE["bg"])
    ax_lat.set_title("Computational Latency  (ms, wall-clock SIL)", fontsize=9, pad=4)
    ax_lat.set_xlabel("Time (s)")
    ax_lat.set_ylabel("Latency (ms)")
    if n > 1:
        ax_lat.plot(d["t_s"], d["ekf_lat_ms"],
                    color=PALETTE["ekf_lat"], lw=0.8, alpha=0.8, label="EKF")
        ax_lat.plot(d["t_s"], d["ctrl_lat_ms"],
                    color=PALETTE["ctrl_lat"], lw=0.8, alpha=0.8, label="MPC")
        # Replan events (non-zero latencies)
        rp_mask = d["replan_lat_ms"] > 0
        if np.any(rp_mask):
            ax_lat.scatter(
                d["t_s"][rp_mask], d["replan_lat_ms"][rp_mask],
                color=PALETTE["replan_lat"], s=12, zorder=5, label="Replanner"
            )
        ax_lat.axhline(20.0, color="#f44", lw=0.6, ls="--", label="20 ms budget")
        if n > 10:
            ax_lat.set_ylim(0, max(d["ctrl_lat_ms"].max() * 1.5, 1.0))
        ax_lat.legend(loc="upper right", fontsize=7, framealpha=0.3, ncol=4)

    # ── Energy Metrics ───────────────────────────────────────────────────
    ax_nrg.clear()
    ax_nrg.set_facecolor(PALETTE["bg"])
    ax_nrg.set_title("Specific Energy Consumption (Wh/km)", fontsize=9, pad=4)
    ax_nrg.set_xlabel("Time (s)")
    ax_nrg.set_ylabel("SEC (Wh/km)")
    if n > 1:
        ax_nrg.plot(d["t_s"], d["sec_wh_km"],
                    color=PALETTE["path"], lw=1.5, label="SEC (Wh/km)")
        ax_nrg.legend(loc="upper right", fontsize=7, framealpha=0.3)
        # Add power info text
        if "power_w" in d:
            ax_nrg.annotate(
                f"Cur Power: {d['power_w'][-1]:.1f} W\nTotal Energy: {d['total_energy_j'][-1]:.0f} J",
                xy=(0.02, 0.85), xycoords='axes fraction', fontsize=8, color="#fff"
            )

    return []


def main():
    p = argparse.ArgumentParser(description="FYP V4 Simulation Dashboard")
    p.add_argument("--csv",     default=str(DEFAULT_CSV), help="Path to telemetry CSV")
    p.add_argument("--refresh", type=float, default=1.0,  help="Refresh interval (s)")
    args = p.parse_args()

    csv_path = pathlib.Path(args.csv)
    print(f"Dashboard: reading {csv_path}  (refresh {args.refresh} s)")
    print("Close the window to exit.")

    setup_figure()
    fig, ax_map, ax_wind, ax_xtrack, ax_goal, ax_lat, ax_nrg = setup_figure()

    ani = FuncAnimation(
        fig,
        update,
        fargs=(csv_path, fig, ax_map, ax_wind, ax_xtrack, ax_goal, ax_lat, ax_nrg),
        interval=int(args.refresh * 1000),
        cache_frame_data=False,
        blit=False,
    )

    plt.show()


if __name__ == "__main__":
    main()
