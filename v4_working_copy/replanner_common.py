"""
Shared infrastructure for Online RH Replanner strategies.

Provides:
    - ReplannerBase: abstract base class all strategies must implement
    - UAVDynamics: shared quadratic drag dynamics model
    - WindField: configurable wind scenarios for benchmarking
    - EnergyModel: energy/power computation from airspeed
    - BenchmarkHarness: runs all strategies across all scenarios and collects metrics

All strategies share the same physics as Module 1 (Wind EKF) and Module 2 (MPC):
    - Quadratic drag: F_drag = k_drag * |v_air|^2 * v_air_hat
    - Power model: P = k_drag * |v_air|^3
    - k_drag = 0.28 (calibrated in Review 1)
"""

import numpy as np
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


# =============================================================================
# Constants (shared with EKF and MPC modules)
# =============================================================================
K_DRAG = 0.28           # Calibrated drag coefficient from Review 1
DT = 0.02               # 50 Hz control loop (matches MPC)
V_CRUISE = 2.0           # Default cruise speed (m/s)
MASS = 1.5               # Placeholder mass (kg) — same as MPC
U_MAX = 8.0              # Max acceleration (m/s^2) — same as MPC
REPLAN_HORIZON = 30      # Default replanning horizon (steps)
REPLAN_DT = 0.02         # Replanning timestep (matches MPC)


# =============================================================================
# Energy Model
# =============================================================================
class EnergyModel:
    """Quadratic drag energy model shared across all modules.

    Power = k_drag * |v_air|^3
    Energy over dt = Power * dt
    """

    def __init__(self, KDrag: float = K_DRAG):
        self.KDrag = KDrag

    def airspeed(self, VGround: np.ndarray, Wind: np.ndarray) -> float:
        """Compute airspeed magnitude."""
        VAir = VGround - Wind
        return float(np.linalg.norm(VAir))

    def power(self, VGround: np.ndarray, Wind: np.ndarray) -> float:
        """Instantaneous power consumption (Watts proxy)."""
        VAir = VGround - Wind
        Speed = np.linalg.norm(VAir)
        return self.KDrag * Speed ** 3

    def energy_segment(self, VGround: np.ndarray, Wind: np.ndarray, Dt: float) -> float:
        """Energy consumed over a time segment (Joules proxy)."""
        return self.power(VGround, Wind) * Dt

    def energy_trajectory(self, Positions: np.ndarray, Velocities: np.ndarray,
                          Wind: np.ndarray, Dt: float) -> float:
        """Total energy for a trajectory. Velocities shape (N,2), Wind shape (2,)."""
        TotalEnergy = 0.0
        for t in range(len(Velocities)):
            TotalEnergy += self.energy_segment(Velocities[t], Wind, Dt)
        return TotalEnergy

    def specific_power(self, Heading: float, Speed: float, Wind: np.ndarray) -> float:
        """Power for flying at a given heading and ground speed."""
        VGround = Speed * np.array([np.cos(Heading), np.sin(Heading)])
        return self.power(VGround, Wind)

    def power_per_meter(self, Heading: float, Speed: float, Wind: np.ndarray) -> float:
        """Energy cost per meter of travel at a given heading."""
        P = self.specific_power(Heading, Speed, Wind)
        VAir = Speed * np.array([np.cos(Heading), np.sin(Heading)]) - Wind
        GroundSpeed = Speed  # Approximate
        if GroundSpeed < 1e-6:
            return float('inf')
        return P / GroundSpeed


# =============================================================================
# UAV Dynamics Model
# =============================================================================
class UAVDynamics:
    """Simple 2D point-mass UAV dynamics with quadratic drag.

    State: [x, y, vx, vy]
    Control: [ax, ay] (acceleration command)
    Wind: [wx, wy] (ambient wind vector)

    Dynamics:
        x_dot = vx
        y_dot = vy
        vx_dot = ax - k_drag * |v_air| * v_air_x
        vy_dot = ay - k_drag * |v_air| * v_air_y

    where v_air = v_ground - wind
    """

    def __init__(self, KDrag: float = K_DRAG, Mass: float = MASS, UMax: float = U_MAX):
        self.KDrag = KDrag
        self.Mass = Mass
        self.UMax = UMax

    def step(self, State: np.ndarray, Control: np.ndarray,
             Wind: np.ndarray, Dt: float = DT) -> np.ndarray:
        """Euler-integrate one timestep."""
        X, Y, Vx, Vy = State
        Ax, Ay = np.clip(Control, -self.UMax, self.UMax)

        # Airspeed
        VairX = Vx - Wind[0]
        VairY = Vy - Wind[1]
        AirSpeed = np.sqrt(VairX**2 + VairY**2) + 1e-8

        # Drag acceleration (opposing airspeed direction)
        DragX = -self.KDrag * AirSpeed * VairX
        DragY = -self.KDrag * AirSpeed * VairY

        # Total acceleration
        VxNew = Vx + (Ax + DragX) * Dt
        VyNew = Vy + (Ay + DragY) * Dt
        XNew = X + Vx * Dt
        YNew = Y + Vy * Dt

        return np.array([XNew, YNew, VxNew, VyNew])

    def rollout(self, State0: np.ndarray, Controls: np.ndarray,
                Wind: np.ndarray, Dt: float = DT) -> np.ndarray:
        """Roll out a control sequence. Returns trajectory shape (H+1, 4)."""
        H = len(Controls)
        Traj = np.zeros((H + 1, 4))
        Traj[0] = State0
        for t in range(H):
            Traj[t + 1] = self.step(Traj[t], Controls[t], Wind, Dt)
        return Traj


# =============================================================================
# Wind Field Models (8 benchmark scenarios)
# =============================================================================
class WindField:
    """Configurable wind field for benchmarking.

    Supports constant, gradient, gusty (Gauss-Markov), and rotating wind.
    """

    def __init__(self, Mode: str = "none", **Params):
        self.Mode = Mode
        self.Params = Params
        self._rng = np.random.default_rng(Params.get("Seed", 42))
        self._t = 0.0

        # Gauss-Markov state for gusty mode
        if Mode == "gusty":
            self._GustState = np.array([0.0, 0.0])

    def get_wind(self, Position: np.ndarray = None, Time: float = None) -> np.ndarray:
        """Return wind vector [wx, wy] at given position/time."""
        if Time is not None:
            self._t = Time

        if self.Mode == "none":
            return np.array([0.0, 0.0])

        elif self.Mode == "constant":
            return np.array(self.Params.get("Wind", [0.0, 0.0]), dtype=float)

        elif self.Mode == "gradient":
            # Linear wind shear: wx depends on y
            Slope = self.Params.get("Slope", 0.3)
            if Position is not None:
                return np.array([Slope * Position[1], 0.0])
            return np.array([0.0, 0.0])

        elif self.Mode == "gusty":
            # Gauss-Markov gust model (same as EKF's synthetic data)
            Tau = self.Params.get("Tau", 8.0)
            Sigma = self.Params.get("Sigma", 2.0)
            Dt = self.Params.get("Dt", DT)
            Decay = 1.0 - Dt / Tau
            Noise = self._rng.normal(0, np.sqrt(0.5 * Dt), 2) * Sigma
            self._GustState = Decay * self._GustState + Noise
            return self._GustState.copy()

        elif self.Mode == "rotating":
            # Rotating wind vector
            Magnitude = self.Params.get("Magnitude", 3.0)
            Omega = self.Params.get("Omega", 0.2)
            Angle = Omega * self._t
            return Magnitude * np.array([np.cos(Angle), np.sin(Angle)])

        else:
            return np.array([0.0, 0.0])

    def reset(self):
        """Reset internal state for a new episode."""
        self._t = 0.0
        self._rng = np.random.default_rng(self.Params.get("Seed", 42))
        if self.Mode == "gusty":
            self._GustState = np.array([0.0, 0.0])


def create_benchmark_scenarios() -> Dict[str, WindField]:
    """Create the 8 standard benchmark wind scenarios."""
    return {
        "S1_NoWind": WindField("none"),
        "S2_Tailwind": WindField("constant", Wind=[3.0, 0.0]),
        "S3_Headwind": WindField("constant", Wind=[-3.0, 0.0]),
        "S4_Crosswind": WindField("constant", Wind=[0.0, 3.0]),
        "S5_Gradient": WindField("gradient", Slope=0.3),
        "S6_Gusty": WindField("gusty", Tau=8.0, Sigma=2.0, Seed=42),
        "S7_Rotating": WindField("rotating", Magnitude=3.0, Omega=0.2),
        "S8_ExtremeHW": WindField("constant", Wind=[-5.0, 0.0]),
    }


# =============================================================================
# Replanner Base Class
# =============================================================================
class ReplannerBase(ABC):
    """Abstract base class for all RH replanner strategies.

    Every strategy must implement:
        - replan(): generate a reference trajectory given current state, goal, and wind
        - name: property returning the strategy name

    The output reference trajectory is consumed by the MPC (Module 2).
    """

    def __init__(self, Config: dict = None):
        self.Config = Config or {}
        self.Horizon = self.Config.get("Horizon", REPLAN_HORIZON)
        self.Dt = self.Config.get("Dt", REPLAN_DT)
        self.VCruise = self.Config.get("VCruise", V_CRUISE)
        self.KDrag = self.Config.get("KDrag", K_DRAG)
        self.EnergyModel = EnergyModel(self.KDrag)
        self.Dynamics = UAVDynamics(self.KDrag)

    @abstractmethod
    def replan(self, State: np.ndarray, Goal: np.ndarray,
               Wind: np.ndarray) -> np.ndarray:
        """Generate reference trajectory for MPC.

        Args:
            State: [x, y, vx, vy] — current UAV state
            Goal: [x_goal, y_goal] — target position
            Wind: [wx, wy] — estimated wind vector from EKF

        Returns:
            RefTraj: shape (Horizon, 4) — reference [x, y, vx, vy] per timestep
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for reporting."""
        raise NotImplementedError

    def _make_straight_line_ref(self, State: np.ndarray, Goal: np.ndarray) -> np.ndarray:
        """Utility: generate a straight-line reference at cruise speed toward goal."""
        Pos = State[:2]
        Dir = Goal - Pos
        Dist = np.linalg.norm(Dir)
        if Dist < 1e-6:
            # Already at goal — hold position
            Ref = np.tile(np.concatenate([Pos, [0.0, 0.0]]), (self.Horizon, 1))
            return Ref

        DirUnit = Dir / Dist
        Vel = DirUnit * self.VCruise

        Ref = np.zeros((self.Horizon, 4))
        for t in range(self.Horizon):
            Progress = min((t + 1) * self.Dt * self.VCruise, Dist)
            Ref[t, :2] = Pos + DirUnit * Progress
            if Progress < Dist:
                Ref[t, 2:4] = Vel
            else:
                Ref[t, :2] = Goal
                Ref[t, 2:4] = [0.0, 0.0]
        return Ref


# =============================================================================
# Benchmark Harness
# =============================================================================
@dataclass
class BenchmarkResult:
    """Results from benchmarking a single strategy on a single scenario."""
    StrategyName: str
    ScenarioName: str
    TotalEnergy: float          # Joules proxy
    MissionTime: float          # seconds
    PathLength: float           # meters
    DirectPathLength: float     # meters (straight line)
    PathLengthRatio: float      # PathLength / DirectPathLength
    EnergySavingsVsDirect: float  # % saved vs straight-line path
    MeanReplanTimeMs: float     # mean replan compute time in ms
    MaxReplanTimeMs: float      # max replan compute time in ms
    GoalReached: bool           # did the UAV reach the goal?
    FinalDistToGoal: float      # distance to goal at end
    TrackingRMSE: float = 0.0   # if MPC tracking is tested
    Trajectory: np.ndarray = field(default_factory=lambda: np.array([]))


class BenchmarkHarness:
    """Runs all replanner strategies across all wind scenarios and collects metrics."""

    def __init__(self, Strategies: List[ReplannerBase] = None,
                 Scenarios: Dict[str, WindField] = None,
                 StartState: np.ndarray = None,
                 Goal: np.ndarray = None,
                 MaxSteps: int = 3000,
                 GoalTolerance: float = 1.0):
        self.Strategies = Strategies or []
        self.Scenarios = Scenarios or create_benchmark_scenarios()
        self.StartState = StartState if StartState is not None else np.array([0.0, 0.0, 0.0, 0.0])
        self.Goal = Goal if Goal is not None else np.array([50.0, 0.0])
        self.MaxSteps = MaxSteps
        self.GoalTolerance = GoalTolerance
        self.Dynamics = UAVDynamics()
        self.Energy = EnergyModel()
        self.Results: List[BenchmarkResult] = []

    def run_single(self, Strategy: ReplannerBase, ScenarioName: str,
                   WindFieldObj: WindField) -> BenchmarkResult:
        """Run a single strategy on a single scenario."""
        WindFieldObj.reset()
        State = self.StartState.copy()
        DirectDist = np.linalg.norm(self.Goal - State[:2])

        TotalEnergy = 0.0
        PathLength = 0.0
        ReplanTimes = []
        Positions = [State[:2].copy()]
        GoalReached = False

        for Step in range(self.MaxSteps):
            Time = Step * DT
            Wind = WindFieldObj.get_wind(State[:2], Time)

            # Replan at 2 Hz (every 25 steps if DT=0.02)
            if Step % 25 == 0:
                T0 = time.perf_counter()
                RefTraj = Strategy.replan(State, self.Goal, Wind)
                ReplanMs = (time.perf_counter() - T0) * 1000.0
                ReplanTimes.append(ReplanMs)

            # Simple proportional controller to track the first reference point
            # (In full pipeline, this would be the MPC)
            if len(RefTraj) > 0:
                Idx = min(Step % 25, len(RefTraj) - 1)
                TargetPos = RefTraj[Idx, :2]
                TargetVel = RefTraj[Idx, 2:4]
                PosError = TargetPos - State[:2]
                VelError = TargetVel - State[2:4]
                Control = 3.0 * PosError + 1.5 * VelError  # PD controller
                Control = np.clip(Control, -U_MAX, U_MAX)
            else:
                Control = np.array([0.0, 0.0])

            # Energy
            VGround = State[2:4]
            TotalEnergy += self.Energy.energy_segment(VGround, Wind, DT)

            # Step dynamics
            PrevPos = State[:2].copy()
            State = self.Dynamics.step(State, Control, Wind, DT)
            PathLength += np.linalg.norm(State[:2] - PrevPos)
            Positions.append(State[:2].copy())

            # Check goal
            if np.linalg.norm(State[:2] - self.Goal) < self.GoalTolerance:
                GoalReached = True
                break

        # Compute direct-path energy (straight line at cruise speed, no replanning)
        WindFieldObj.reset()
        DirectEnergy = 0.0
        DirectState = self.StartState.copy()
        DirectDir = (self.Goal - DirectState[:2])
        DirectDirUnit = DirectDir / (np.linalg.norm(DirectDir) + 1e-8)
        DirectVel = DirectDirUnit * V_CRUISE
        for Step in range(self.MaxSteps):
            Wind = WindFieldObj.get_wind(DirectState[:2], Step * DT)
            DirectEnergy += self.Energy.energy_segment(DirectVel, Wind, DT)
            DirectState[:2] += DirectVel * DT
            if np.linalg.norm(DirectState[:2] - self.Goal) < self.GoalTolerance:
                break

        MissionTime = (Step + 1) * DT
        EnergySavings = (DirectEnergy - TotalEnergy) / max(DirectEnergy, 1e-8) * 100.0
        PathRatio = PathLength / max(DirectDist, 1e-8)

        return BenchmarkResult(
            StrategyName=Strategy.name,
            ScenarioName=ScenarioName,
            TotalEnergy=TotalEnergy,
            MissionTime=MissionTime,
            PathLength=PathLength,
            DirectPathLength=DirectDist,
            PathLengthRatio=PathRatio,
            EnergySavingsVsDirect=EnergySavings,
            MeanReplanTimeMs=float(np.mean(ReplanTimes)) if ReplanTimes else 0.0,
            MaxReplanTimeMs=float(np.max(ReplanTimes)) if ReplanTimes else 0.0,
            GoalReached=GoalReached,
            FinalDistToGoal=float(np.linalg.norm(State[:2] - self.Goal)),
            Trajectory=np.array(Positions),
        )

    def run_all(self) -> List[BenchmarkResult]:
        """Run all strategies on all scenarios."""
        self.Results = []
        Total = len(self.Strategies) * len(self.Scenarios)
        Count = 0
        for Strategy in self.Strategies:
            for ScenarioName, WindFieldObj in self.Scenarios.items():
                Count += 1
                print(f"  [{Count}/{Total}] {Strategy.name} × {ScenarioName}...", end=" ")
                Result = self.run_single(Strategy, ScenarioName, WindFieldObj)
                self.Results.append(Result)
                Status = "OK GOAL" if Result.GoalReached else "XX MISS"
                print(f"{Status} | Energy={Result.TotalEnergy:.1f} | "
                      f"Savings={Result.EnergySavingsVsDirect:+.1f}% | "
                      f"Replan={Result.MeanReplanTimeMs:.2f}ms")
        return self.Results

    def summary_table(self) -> str:
        """Generate a formatted summary table of all results."""
        Lines = []
        Lines.append(f"{'Strategy':<25} {'Scenario':<15} {'Energy':>10} {'Savings':>10} "
                      f"{'Time(s)':>8} {'PathRatio':>10} {'Replan(ms)':>11} {'Goal':>6}")
        Lines.append("-" * 105)
        for R in self.Results:
            Goal = "YES" if R.GoalReached else "NO"
            Lines.append(f"{R.StrategyName:<25} {R.ScenarioName:<15} {R.TotalEnergy:>10.1f} "
                          f"{R.EnergySavingsVsDirect:>+9.1f}% {R.MissionTime:>8.2f} "
                          f"{R.PathLengthRatio:>10.3f} {R.MeanReplanTimeMs:>11.3f} {Goal:>6}")
        return "\n".join(Lines)
