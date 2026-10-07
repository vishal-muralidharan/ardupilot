#!/usr/bin/env bash
# ============================================================================
#  FYP V4 Gazebo Simulation – Injambakkam, Chennai
# ============================================================================
#
#  This script starts the full simulation stack:
#    1. ArduPilot SITL (ArduCopter, GPS-spoofed to Injambakkam)
#    2. Gazebo Harmonic (injambakkam_fyp world with Iris + ArduPilot plugin)
#    3. FYP V4 MAVLink Bridge (WindEKF + EWMPC + MPPI/Greedy replanner)
#
#  Usage:
#    ./launch_simulation.sh [mppi|greedy] [altitude_m] [goal_north_m]
#
#  Examples:
#    ./launch_simulation.sh                  # MPPI, 30 m, 1000 m north
#    ./launch_simulation.sh greedy 40 800    # Greedy, 40 m AGL, 800 m north
#    ./launch_simulation.sh mppi 30 1000     # Explicit defaults
#
#  Prerequisites:
#    • Gazebo Harmonic (gz sim)        installed at /opt/homebrew/bin/gz
#    • ardupilot_gazebo plugin         built in ~/Applications/Gazebo/ardupilot_gazebo/build
#    • ArduPilot SITL (sim_vehicle.py) at ~/Applications/Gazebo/ardupilot/Tools/autotest
#    • Python 3.10 with pymavlink      at ~/.pyenv/versions/3.10.18/bin/python
#    • numpy                           installed in the Python 3.10 env
#
# ============================================================================

set -euo pipefail

export PATH="/Users/vishalmuralidharan/.pyenv/versions/3.10.18/bin:$PATH"

# ─────────────────────────────────────────────────────────────────────────────
# Argument Parsing
# ─────────────────────────────────────────────────────────────────────────────
REPLANNER="${1:-mppi}"
ALTITUDE="${2:-30}"
GOAL_NORTH="${3:-1000}"

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  FYP V4 Gazebo Simulation  –  Injambakkam, Chennai              ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  Location  : 12.9516°N  80.2573°E  (Injambakkam Beach)         ║"
echo "║  Replanner : ${REPLANNER}                                                ║"
echo "║  Altitude  : ${ALTITUDE} m AGL                                         ║"
echo "║  Goal      : ${GOAL_NORTH} m North                                      ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GAZEBO_DIR="${SCRIPT_DIR}"
ARDUPILOT_DIR="${GAZEBO_DIR}/ardupilot"
PLUGIN_DIR="${GAZEBO_DIR}/ardupilot_gazebo"
WORLDS_DIR="${PLUGIN_DIR}/worlds"
MODELS_DIR="${PLUGIN_DIR}/models"
BUILD_DIR="${PLUGIN_DIR}/build"

PYTHON310="/Users/vishalmuralidharan/.pyenv/versions/3.10.18/bin/python"
SIM_VEHICLE="${ARDUPILOT_DIR}/Tools/autotest/sim_vehicle.py"
BRIDGE_SCRIPT="${GAZEBO_DIR}/fyp_v4_mavlink_bridge.py"
WORLD_FILE="${WORLDS_DIR}/injambakkam_fyp.sdf"
PARAMS_FILE="${GAZEBO_DIR}/injambakkam_ardupilot.parm"

LOG_DIR="${GAZEBO_DIR}/sim_logs"
mkdir -p "${LOG_DIR}"

# ─────────────────────────────────────────────────────────────────────────────
# Verify prerequisites
# ─────────────────────────────────────────────────────────────────────────────
echo "[1/5] Checking prerequisites …"

if ! command -v gz &>/dev/null; then
    echo "❌  Gazebo 'gz' not found. Install Gazebo Harmonic and retry."
    exit 1
fi

if [ ! -f "${SIM_VEHICLE}" ]; then
    echo "❌  ArduPilot sim_vehicle.py not found at: ${SIM_VEHICLE}"
    echo "    Clone ArduPilot: git clone https://github.com/ArduPilot/ardupilot.git"
    exit 1
fi

if [ ! -f "${PYTHON310}" ]; then
    echo "❌  Python 3.10 not found at: ${PYTHON310}"
    exit 1
fi

if [ ! -f "${BRIDGE_SCRIPT}" ]; then
    echo "❌  Bridge script not found: ${BRIDGE_SCRIPT}"
    exit 1
fi

if [ ! -f "${WORLD_FILE}" ]; then
    echo "❌  Gazebo world file not found: ${WORLD_FILE}"
    exit 1
fi

echo "    ✅  All prerequisites found"

# ─────────────────────────────────────────────────────────────────────────────
# Gazebo Environment Variables
# ─────────────────────────────────────────────────────────────────────────────
echo "[2/5] Setting up Gazebo environment …"

export GZ_SIM_RESOURCE_PATH="${MODELS_DIR}:/opt/homebrew/share/gz/gz-sim8/worlds:/opt/homebrew/share/gz/gz-common5/media/materials/textures"
export GZ_SIM_SYSTEM_PLUGIN_PATH="${BUILD_DIR}:/opt/homebrew/lib/gz-sim8/plugins"
export GZ_VERSION=harmonic

# Check if plugin was built
if [ ! -d "${BUILD_DIR}" ]; then
    echo "⚠️  ardupilot_gazebo build directory not found. Building now …"
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    cmake "${PLUGIN_DIR}" -DCMAKE_BUILD_TYPE=RelWithDebInfo 2>&1 | tail -5
    make -j4 2>&1 | tail -10
    cd "${GAZEBO_DIR}"
fi

echo "    ✅  GZ_SIM_RESOURCE_PATH=${GZ_SIM_RESOURCE_PATH}"
echo "    ✅  GZ_SIM_SYSTEM_PLUGIN_PATH=${GZ_SIM_SYSTEM_PLUGIN_PATH}"

# ─────────────────────────────────────────────────────────────────────────────
# Function: Cleanup on exit
# ─────────────────────────────────────────────────────────────────────────────
SITL_PID=""
GAZEBO_PID=""
BRIDGE_PID=""

cleanup() {
    echo ""
    echo "[CLEANUP] Stopping all processes …"
    [ -n "${BRIDGE_PID}" ] && kill "${BRIDGE_PID}" 2>/dev/null && echo "    Stopped bridge   (PID ${BRIDGE_PID})"
    [ -n "${GAZEBO_PID}" ] && kill "${GAZEBO_PID}" 2>/dev/null && echo "    Stopped Gazebo   (PID ${GAZEBO_PID})"
    [ -n "${SITL_PID}"   ] && kill "${SITL_PID}"   2>/dev/null && echo "    Stopped SITL     (PID ${SITL_PID})"
    sleep 2
    # Kill any remaining gz or ardupilot processes
    pkill -f "gz sim"            2>/dev/null || true
    pkill -f "arducopter"        2>/dev/null || true
    pkill -f "fyp_v4_mavlink"    2>/dev/null || true
    echo "    ✅  Cleanup complete"
}
trap cleanup EXIT INT TERM

# ─────────────────────────────────────────────────────────────────────────────
# [3/5] Start ArduPilot SITL
# ─────────────────────────────────────────────────────────────────────────────
echo "[3/5] Starting ArduPilot SITL (ArduCopter @ Injambakkam) …"
echo "      GPS: 12.9516°N 80.2573°E  Alt: 6 m MSL"

SITL_LOG="${LOG_DIR}/sitl_$(date +%Y%m%d_%H%M%S).log"

# Run SITL from a temporary working directory
SITL_WORKDIR="/tmp/sitl_injambakkam"
mkdir -p "${SITL_WORKDIR}"

ARDUCOPTER_BIN="${GAZEBO_DIR}/ardupilot/build/sitl/bin/arducopter"
"${ARDUCOPTER_BIN}" \
    -w \
    --model JSON \
    --speedup 1 \
    --slave 0 \
    --defaults "${PARAMS_FILE}" \
    --sim-address 127.0.0.1 \
    -I0 \
    --home "12.9516,80.2573,6.0,0.0" \
    >"${SITL_LOG}" 2>&1 &

SITL_PID=$!
echo "      SITL started  PID=${SITL_PID}  log=${SITL_LOG}"

# Wait for SITL to initialise
echo "      Waiting 12 s for SITL to initialise …"
sleep 12

if ! kill -0 "${SITL_PID}" 2>/dev/null; then
    echo "❌  SITL failed to start. Check log: ${SITL_LOG}"
    exit 1
fi
echo "    ✅  SITL running"

# ─────────────────────────────────────────────────────────────────────────────
# [4/5] Start Gazebo
# ─────────────────────────────────────────────────────────────────────────────
echo "[4/5] Starting Gazebo Harmonic with Injambakkam world …"

GAZEBO_LOG="${LOG_DIR}/gazebo_$(date +%Y%m%d_%H%M%S).log"

gz sim -s -r "${WORLD_FILE}" >"${GAZEBO_LOG}" 2>&1 &
GAZEBO_PID=$!
echo "      Gazebo started  PID=${GAZEBO_PID}  log=${GAZEBO_LOG}"

# Wait for Gazebo to fully start
echo "      Waiting 15 s for Gazebo to load world …"
sleep 15

if ! kill -0 "${GAZEBO_PID}" 2>/dev/null; then
    echo "❌  Gazebo failed to start. Check log: ${GAZEBO_LOG}"
    exit 1
fi
echo "    ✅  Gazebo running"

# ─────────────────────────────────────────────────────────────────────────────
# [5/5] Start FYP V4 MAVLink Bridge
# ─────────────────────────────────────────────────────────────────────────────
echo "[5/5] Starting FYP V4 MAVLink Bridge …"
echo "      Replanner : ${REPLANNER}"
echo "      Altitude  : ${ALTITUDE} m"
echo "      Goal      : ${GOAL_NORTH} m North"

BRIDGE_LOG="${LOG_DIR}/bridge_$(date +%Y%m%d_%H%M%S).log"

# Give MAVLink time to establish connection
sleep 3

${PYTHON310} "${BRIDGE_SCRIPT}" \
    --replanner "${REPLANNER}" \
    --goal-north "${GOAL_NORTH}" \
    --altitude "${ALTITUDE}" \
    >"${BRIDGE_LOG}" 2>&1 &

BRIDGE_PID=$!
echo "      Bridge started  PID=${BRIDGE_PID}  log=${BRIDGE_LOG}"
echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  Simulation is RUNNING  –  press Ctrl+C to stop"
echo "══════════════════════════════════════════════════════════════════"
echo "  Live bridge log:   tail -f ${BRIDGE_LOG}"
echo "  Live SITL log:     tail -f ${SITL_LOG}"
echo "  Telemetry CSV:     ${GAZEBO_DIR}/fyp_v4_telemetry.csv"
echo "══════════════════════════════════════════════════════════════════"
echo ""

# Wait for bridge to finish (or Ctrl+C)
wait "${BRIDGE_PID}" || true

echo ""
echo "Bridge finished. Stopping simulation …"
