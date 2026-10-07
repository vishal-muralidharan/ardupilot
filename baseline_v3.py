# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION AND ASSUMPTIONS
# ─────────────────────────────────────────────────────────────────────────────
import pathlib

QuickTest = False
StartPos = (0.0, 0.0)
GoalPos = (1000.0, 0.0)
BaseWind = (-1.5, 1.2)
ShearSlope = 0.008
EnvSeed = 42
FlightSteps = 200 if QuickTest else 2000
Dt = 0.02
GoalTol = 5.0

BusVoltageV = 14.8
BatteryCapacityWh = 74.0
MppiConfig = dict(NumRollouts=256, Temperature=10.0, NoiseSigma=1.5, EnergyWeight=0.5)
GreedyConfig = dict(BlendWeight=0.3, MaxDeviation=10.0)

CachePath = pathlib.Path("turbulence_comparison_cache.json")
print(f"CONFIGURATION | STEPS={FlightSteps} | DT={Dt}S | ASSUMPTIONS: BUS={BusVoltageV}V BAT={BatteryCapacityWh}WH")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0: FILE MANIFEST & SHA-256
# ─────────────────────────────────────────────────────────────────────────────
import hashlib
import pathlib
import sys

NbDir = pathlib.Path.cwd()
FypRoot = NbDir.parents[3]

Manifest = {
    "current_ekf (WindEKFV6 inside Integrated Pipeline.py)": FypRoot / "Second Review/Code/Integration/Version 1/Integrated Pipeline.py",
    "telemetry_ekf_placeholder (EMPTY 0 bytes)": FypRoot / "Second Review/Code/Integration/Version 1/Telemetry EKF.py",
    "wind_ekf_class_v6": FypRoot / "First Review/Code/Version 6/Wind EKF Class V6.py",
    "baseline_ekf_v1 (joint drag/wind, linear, 3-state)": FypRoot / "First Review/Code/Version 1/src/ekf.py",
    "baseline_ekf_normalekf (Cell26, same structure as V1)": FypRoot / "First Review/Code/Version 6/Cell 26 NormalEKF Class.py",
    "adaptive_ltv_mpc": FypRoot / "Second Review/Code/EWMPC/Version 2/Controllers/Adaptive LTV MPC.py",
    "single_shooting_slsqp": FypRoot / "Second Review/Code/EWMPC/Version 2/Controllers/Single Shooting.py",
    "mppi_v1": FypRoot / "Second Review/Code/Replanner/Strategy 3 - MPPI/Version 1/MPPI - V1.py",
    "greedy_v1": FypRoot / "Second Review/Code/Replanner/Strategy 6 - Greedy/Version 1/Greedy - V1.py",
    "replanner_common": FypRoot / "Second Review/Code/Replanner/Common/Replanner Common.py",
    "ewmpc_dynamics": FypRoot / "Second Review/Code/EWMPC/Version 2/Dynamics.py",
}

def GetSha16(PathObj):
    P = pathlib.Path(PathObj)
    if not P.exists():
        return "FILE_NOT_FOUND"
    if P.stat().st_size == 0:
        return "EMPTY_0_BYTES"
    return hashlib.sha256(P.read_bytes()).hexdigest()[:16]

print(f"{'KEY':<55} {'SHA-256[:16]':<18} {'BYTES':<10} NOTES")
print("-" * 110)
for Label, PathObj in Manifest.items():
    P = pathlib.Path(PathObj)
    Size = P.stat().st_size if P.exists() else -1
    S = GetSha16(PathObj)
    Note = "WARNING EMPTY" if S == "EMPTY_0_BYTES" else ("WARNING MISSING" if S == "FILE_NOT_FOUND" else "")
    print(f"{Label:<55} {S:<18} {Size:<10} {Note}")

print("\nKEY FINDINGS:")
print("  WindEKFV6: EMBEDDED IN 'Integrated Pipeline.py' (CLASS WindEKFV6)")
print("  Telemetry EKF.py: 0 BYTES, PLACEHOLDER ONLY")
print("  Baseline EKF: V1/src/ekf.py - 3-STATE [wx,wy,k_drag], LINEAR DRAG, JOINT ESTIMATION")
print("  NormalEKF (Cell 26): IDENTICAL STRUCTURE TO V1; V1 IS USED AS BASELINE")


# ─────────────────────────────────────────────────────────────────────────────
# WORKING-COPY BUILD (SNAKE_CASE COPIES; ORIGINALS NEVER MODIFIED)
# ─────────────────────────────────────────────────────────────────────────────
import shutil
import sys
import pathlib

NbDir = pathlib.Path.cwd()
FypRoot = NbDir.parents[3]
WorkDir = NbDir / "Working Copy"
WorkDir.mkdir(exist_ok=True)

Copies = {
    "integrated_pipeline": FypRoot / "Second Review/Code/Integration/Version 1/Integrated Pipeline.py",
    "baseline_ekf_v1": FypRoot / "First Review/Code/Version 1/src/ekf.py",
    "adaptive_ltv_mpc": FypRoot / "Second Review/Code/EWMPC/Version 2/Controllers/Adaptive LTV MPC.py",
    "single_shooting_slsqp": FypRoot / "Second Review/Code/EWMPC/Version 2/Controllers/Single Shooting.py",
    "mppi_v1": FypRoot / "Second Review/Code/Replanner/Strategy 3 - MPPI/Version 1/MPPI - V1.py",
    "greedy_v1": FypRoot / "Second Review/Code/Replanner/Strategy 6 - Greedy/Version 1/Greedy - V1.py",
    "replanner_common": FypRoot / "Second Review/Code/Replanner/Common/Replanner Common.py",
    "ewmpc_dynamics": FypRoot / "Second Review/Code/EWMPC/Version 2/Dynamics.py",
}

for Snake, Src in Copies.items():
    Dst = WorkDir / f"{Snake}.py"
    shutil.copy2(Src, Dst)
    if Snake in ('adaptive_ltv_mpc', 'single_shooting_slsqp'):
        Txt = Dst.read_text()
        Dst.write_text(Txt.replace('from .base import', 'from base import'))
    print(f"  {Src.name} -> {Dst.name}")

if str(WorkDir) not in sys.path:
    sys.path.insert(0, str(WorkDir))
print("Working Copy/ ON SYS.PATH. ORIGINALS UNTOUCHED.")


# ─────────────────────────────────────────────────────────────────────────────
# IMPORT CLASSES FROM WORKING COPIES
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import types
import abc
import sys

# 1. Pipeline
import integrated_pipeline as ip
for Attr in ["WindEKFV6", "UnifiedFlightPipeline", "WindSeerEnvironment", "UAVPlantDynamics"]:
    assert hasattr(ip, Attr), f"{Attr} NOT FOUND IN integrated_pipeline"
print("integrated_pipeline: ALL 4 REQUIRED CLASSES FOUND")

# Verify WindEKFV6 API
EkfInst = ip.WindEKFV6(KDragFixed=0.28)
EkfInst.reset()
W = EkfInst.step(0.02, np.array([1.0, 0.0]), np.array([0.1, 0.0]), np.array([0.05, 0.0]))
assert W.shape == (2,), "WindEKFV6.step MUST RETURN (2,)"
print(f"WindEKFV6.step() CONFIRMED: {W}")

# 2. Baseline EKF
import baseline_ekf_v1 as bev1
BaselineWindEKF = bev1.WindEKF
if not hasattr(BaselineWindEKF, "reset"):
    class BekfReset(BaselineWindEKF):
        def reset(self, initial_wind=None):
            self.x = np.array([0.0, 0.0, 0.30])
            self.P = np.diag([4.0, 4.0, 0.05]).astype(float)
    BaselineWindEKF = BekfReset
    print("ADDED reset() TO BaselineWindEKF (NOT IN ORIGINAL V1 ekf.py)")

BInst = BaselineWindEKF()
BInst.reset()
W2 = BInst.step(0.02, np.array([1.0, 0.0]), np.array([0.1, 0.0]), np.array([0.05, 0.0]))
print(f"BaselineWindEKF (V1, LINEAR, JOINT K_DRAG): STEP -> {W2}")

# 3. EWMPC V2 Dynamics + controllers
import ewmpc_dynamics as ewdyn
KDragPlantEffective = 0.28 / 1.5
KDragNominal = 0.28
print(f"ewmpc_dynamics: K_DRAG_PLANT_EFFECTIVE={KDragPlantEffective:.5f}")

sys.modules["dynamics"] = ewdyn
Bm = types.ModuleType("base")
class Bc(abc.ABC):
    def __init__(self, name):
        self.name = name
    @abc.abstractmethod
    def reset(self):
        pass
    @abc.abstractmethod
    def compute_control(self, state, target, wind_magnitude, wind_vector=None, dt=0.1):
        pass
Bm.BaseController = Bc
for K in ("base", ".base", "controllers.base"):
    sys.modules[K] = Bm

import adaptive_ltv_mpc as altv
AdaptiveLTVDisturbanceObserverMPC = altv.AdaptiveLTVDisturbanceObserverMPC
print("AdaptiveLTVDisturbanceObserverMPC LOADED")

import single_shooting_slsqp as ss
SingleShootingSLSQP = ss.SingleShootingSLSQP
print("SingleShootingSLSQP LOADED")

# 4. Replanners
import replanner_common as rc
import mppi_v1 as mppi
Strategy3MPPI = mppi.Strategy3MPPI

import greedy_v1 as greedy
Strategy6Greedy = greedy.Strategy6Greedy

St = np.array([0.0, 0.0, 0.0, 0.0])
Gl = np.array([100.0, 0.0])
Wd = np.array([-1.0, 0.5])
T1 = Strategy3MPPI(MppiConfig).replan(St, Gl, Wd)
T2 = Strategy6Greedy(GreedyConfig).replan(St, Gl, Wd)
assert T1.shape[1] == 4 and T2.shape[1] == 4
print(f"MPPI -> {T1.shape}, GREEDY -> {T2.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTER CLASSES
# ─────────────────────────────────────────────────────────────────────────────
class AdaptiveMPCAdapter:
    def __init__(self, dynamics):
        self._ctrl = AdaptiveLTVDisturbanceObserverMPC(dynamics=dynamics)
        self.name = self._ctrl.name

    def reset(self):
        self._ctrl.reset()

    def compute_control(self, state, target, wind_est=None, dt=0.02):
        WMag, WVec = 0.0, None
        if wind_est is not None:
            A = np.asarray(wind_est, float)
            WMag, WVec = float(np.linalg.norm(A)), A
        return self._ctrl.compute_control(state, target, WMag, WVec, dt)

class SLSQPAdapter:
    def __init__(self, dynamics):
        self._ctrl = SingleShootingSLSQP(dynamics=dynamics, horizon=20)
        self.name = self._ctrl.name

    def reset(self):
        self._ctrl.reset()

    def compute_control(self, state, target, wind_est=None, dt=0.02):
        WMag, WVec = 0.0, None
        if wind_est is not None:
            A = np.asarray(wind_est, float)
            WMag, WVec = float(np.linalg.norm(A)), A
        return self._ctrl.compute_control(state, target, WMag, WVec, dt)

class ReplannerAdapter:
    def __init__(self, r):
        self._r = r

    def replan(self, state, goal, wind_estimate=None):
        W = np.zeros(2) if wind_estimate is None else np.asarray(wind_estimate, float)
        return self._r.replan(state, goal, W)

class BaselineEKFAdapter:
    def __init__(self):
        self._ekf = BaselineWindEKF()

    def reset(self, initial_wind=None):
        self._ekf = BaselineWindEKF()

    def step(self, dt, v, at, am):
        return self._ekf.step(dt, v, at, am)

print("ADAPTERS DEFINED: AdaptiveMPCAdapter, SLSQPAdapter, ReplannerAdapter, BaselineEKFAdapter")


# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT AND PIPELINE INSTRUMENTATION
# ─────────────────────────────────────────────────────────────────────────────
import time as time_mod

class WindSeerGustEnvironment(ip.WindSeerEnvironment):
    """Adds 1-cosine gust overlay to WindSeerEnvironment Dryden turbulence."""
    def __init__(self, BaseWind=(-1.5, 1.2), ShearSlope=0.008, Dt=0.02,
                 TurbulenceSigma=0.35, TurbTau=4.0, Seed=42,
                 GustAmplitude=0.0, GustPeriodS=8.0, GustDurationS=2.0):
        super().__init__(BaseWind=BaseWind, ShearSlope=ShearSlope,
                         Dt=Dt, TurbulenceSigma=TurbulenceSigma, Seed=Seed)
        self.Tau = TurbTau
        self.GustAmp = GustAmplitude
        self.GustPeriod = GustPeriodS
        self.GustDur = GustDurationS
        self._step = 0

    def sample_wind(self, pos):
        W = super().sample_wind(pos)
        if self.GustAmp > 0:
            TMod = (self._step * self.Dt) % self.GustPeriod
            if TMod < self.GustDur:
                Phase = TMod / self.GustDur
                W = W + np.array([self.GustAmp * (1.0 - np.cos(2 * np.pi * Phase)) * 0.5, 0.0])
        self._step += 1
        return W

ScenarioDefs = {
    "S1_turbulent": dict(TurbulenceSigma=1.0, TurbTau=4.0, GustAmplitude=0.0, GustPeriodS=8.0, GustDurationS=2.0),
    "S2_gust_prone": dict(TurbulenceSigma=1.5, TurbTau=2.0, GustAmplitude=3.0, GustPeriodS=8.0, GustDurationS=2.0),
    "S3_severe_gusts": dict(TurbulenceSigma=2.0, TurbTau=1.5, GustAmplitude=5.0, GustPeriodS=6.0, GustDurationS=1.5),
}

class InstrumentedPipeline(ip.UnifiedFlightPipeline):
    """Subclass that monkey-patches sample_wind and Ekf.step to capture logs."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._true_wind_log = []
        self._est_wind_log = []

    def execute_mission(self, *args, **kwargs):
        self._true_wind_log = []
        self._est_wind_log = []
        Os = self.Environment.sample_wind
        Oe = self.Ekf.step
        S = self

        def Ps(pos):
            W = Os(pos)
            S._true_wind_log.append(W.copy())
            return W

        def Pe(dt, vg, at, am):
            Wh = Oe(dt, vg, at, am)
            S._est_wind_log.append(np.asarray(Wh).copy())
            return Wh

        self.Environment.sample_wind = Ps
        self.Ekf.step = Pe
        try:
            Result = super().execute_mission(*args, **kwargs)
        finally:
            self.Environment.sample_wind = Os
            self.Ekf.step = Oe
        return Result

print(f"WindSeerGustEnvironment + InstrumentedPipeline DEFINED | SCENARIOS: {list(ScenarioDefs)}")


# ─────────────────────────────────────────────────────────────────────────────
# METRIC HELPERS
# ─────────────────────────────────────────────────────────────────────────────
import math

def ComputeWindRmse(WHat, WTrue):
    if not WHat:
        return 0.0, 0.0
    E = np.array([np.linalg.norm(np.asarray(H) - np.asarray(T)) for H, T in zip(WHat, WTrue)])
    Ss = float(np.sqrt(np.mean(E[max(1, len(E) // 4):] ** 2)))
    return float(np.sqrt(np.mean(E ** 2))), Ss

def ComputeConvergenceTime(WHat, WTrue, Dt, SsRmse, Mult=1.5, Win=10):
    if SsRmse < 1e-9 or not WHat:
        return float("nan")
    E = np.array([np.linalg.norm(np.asarray(H) - np.asarray(T)) for H, T in zip(WHat, WTrue)])
    Cnt = 0
    for I, B in enumerate(E < Mult * SsRmse):
        Cnt = Cnt + 1 if B else 0
        if Cnt >= Win:
            return float((I - Win + 1) * Dt * 1000.0)
    return float("nan")

def ComputeSec(Ej, Pm, Aej=None):
    Km = max(Pm, 1.0) / 1000.0
    return (Ej / 3600.0) / Km, ((Aej or 0.0) / 3600.0) / Km

def ComputeMotorCurrent(Ej, Bv=BusVoltageV):
    return Ej / max(Bv, 1e-6)

def ComputeEndurance(Pw, Bwh=BatteryCapacityWh):
    return float("nan") if Pw < 1e-3 else (Bwh * 3600.0) / Pw

print("METRIC HELPERS: ComputeWindRmse, ComputeConvergenceTime, ComputeSec, ComputeMotorCurrent, ComputeEndurance")


# ─────────────────────────────────────────────────────────────────────────────
# STACK DEFINITIONS AND EXECUTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
StackNames = ["full", "baseline", "ekf_ablation", "diagnostic"]
StackLabels = {
    "full": "Full (WindEKFV6+Adaptive+MPPI)",
    "baseline": "Baseline (JointEKF+SLSQP+Greedy)",
    "ekf_ablation": "EKF-Ablation (JointEKF+Adaptive+MPPI)",
    "diagnostic": "[DIAG] WindEKFV6 k=0.28/1.5+Adaptive+MPPI",
}

def MakePipeline(Stack, Sp):
    np.random.seed(42)
    Env = WindSeerGustEnvironment(
        BaseWind=BaseWind, ShearSlope=ShearSlope, Dt=Dt,
        TurbulenceSigma=Sp["TurbulenceSigma"], TurbTau=Sp["TurbTau"],
        Seed=EnvSeed, GustAmplitude=Sp["GustAmplitude"],
        GustPeriodS=Sp["GustPeriodS"], GustDurationS=Sp["GustDurationS"]
    )
    Dyn = ip.UAVPlantDynamics(Mass=1.5, KDrag=0.28)
    D = lambda: ewdyn.UAVDynamics(k_drag=KDragPlantEffective, dt=Dt, u_bounds=(-5.0, 5.0))
    
    if Stack == "full":
        Ekf = ip.WindEKFV6(KDragFixed=KDragNominal)
        Ctrl = AdaptiveMPCAdapter(D())
        Repl = ReplannerAdapter(Strategy3MPPI(MppiConfig))
    elif Stack == "baseline":
        Ekf = BaselineEKFAdapter()
        Ctrl = SLSQPAdapter(D())
        Repl = ReplannerAdapter(Strategy6Greedy(GreedyConfig))
    elif Stack == "ekf_ablation":
        Ekf = BaselineEKFAdapter()
        Ctrl = AdaptiveMPCAdapter(D())
        Repl = ReplannerAdapter(Strategy3MPPI(MppiConfig))
    elif Stack == "diagnostic":
        Ekf = ip.WindEKFV6(KDragFixed=KDragPlantEffective)
        Ctrl = AdaptiveMPCAdapter(D())
        Repl = ReplannerAdapter(Strategy3MPPI(MppiConfig))
    else:
        raise ValueError(Stack)
        
    return InstrumentedPipeline(
        Replanner=Repl, Controller=Ctrl, Ekf=Ekf,
        Environment=Env, Dynamics=Dyn,
        ControlDt=Dt, ReplanIntervalSteps=25
    )

def RunSingle(Stack, Scen, Sp):
    np.random.seed(42)
    Pl = MakePipeline(Stack, Sp)
    T0 = time_mod.perf_counter()
    R = Pl.execute_mission(
        start_pos=StartPos, goal_pos=GoalPos,
        max_steps=FlightSteps, goal_tolerance=GoalTol,
        use_ekf_feedback=True
    )
    WallTime = time_mod.perf_counter() - T0
    Wh, Wt = Pl._est_wind_log, Pl._true_wind_log
    N = min(len(Wh), len(Wt))
    Wh, Wt = Wh[:N], Wt[:N]
    
    Rmse, Ss = ComputeWindRmse(Wh, Wt)
    Conv = ComputeConvergenceTime(Wh, Wt, Dt, Ss)
    Ej = R["total_energy_kj"] * 1000.0
    Aej = R["aero_energy_kj"] * 1000.0
    Pm = R["path_length_m"]
    Sec, SecA = ComputeSec(Ej, Pm, Aej)
    
    return {
        "stack": Stack, "scenario": Scen, "goal_reached": R["goal_reached"],
        "mission_time_s": R["mission_time_s"], "wall_time_s": WallTime,
        "wind_rmse": Rmse, "wind_ss_rmse": Ss, "conv_time_ms": Conv,
        "tracking_rmse_m": R["tracking_rmse_m"], "cross_track_rmse_m": R["mean_cross_track_m"],
        "path_length_m": Pm, "total_energy_j": Ej, "aero_energy_j": Aej,
        "mean_power_w": R["mean_power_w"], "sec_wh_km": Sec, "sec_aero_wh_km": SecA,
        "motor_current_as": ComputeMotorCurrent(Ej),
        "endurance_s": ComputeEndurance(R["mean_power_w"]),
        "ekf_lat_mean_ms": R["mean_ekf_lat_ms"], "ekf_lat_p95_ms": R["p95_ekf_lat_ms"], "ekf_lat_max_ms": R["max_ekf_lat_ms"],
        "ctrl_lat_mean_ms": R["mean_ctrl_lat_ms"], "ctrl_lat_p95_ms": R["p95_ctrl_lat_ms"], "ctrl_lat_max_ms": R["max_ctrl_lat_ms"],
        "combined_lat_mean_ms": R["mean_ekf_lat_ms"] + R["mean_ctrl_lat_ms"],
        "combined_lat_p95_ms": R["p95_ekf_lat_ms"] + R["p95_ctrl_lat_ms"],
        "combined_lat_max_ms": R["max_ekf_lat_ms"] + R["max_ctrl_lat_ms"],
        "replan_lat_mean_ms": R["mean_replan_lat_ms"], "replan_lat_max_ms": R["max_replan_lat_ms"],
        "terminal_dist_m": R["terminal_dist_to_goal_m"],
        "wind_hat_history": [list(W) for W in Wh], "wind_true_history": [list(W) for W in Wt],
    }

print("MakePipeline + RunSingle DEFINED")
for K, V in StackLabels.items():
    print(f"  {K}: {V}")


# ─────────────────────────────────────────────────────────────────────────────
# EXECUTE ALL RUNS
# ─────────────────────────────────────────────────────────────────────────────
import json

def GetRunKey(S, Sc):
    return f"{S}__{Sc}"

def LoadCache():
    if CachePath.exists():
        with open(CachePath) as F:
            return json.load(F)
    return {}

def SaveCache(C):
    with open(CachePath, "w") as F:
        json.dump(C, F, indent=2)

Cache = LoadCache()
AllResults = {}
TotalRuns = len(StackNames) * len(ScenarioDefs)
DoneRuns = 0

for Stack in StackNames:
    for Scen, Params in ScenarioDefs.items():
        Key = GetRunKey(Stack, Scen)
        DoneRuns += 1
        if Key in Cache:
            print(f"[{DoneRuns}/{TotalRuns}] {Key} — CACHED")
            AllResults[Key] = Cache[Key]
        else:
            print(f"[{DoneRuns}/{TotalRuns}] {Key} … ", end="", flush=True)
            T0 = time_mod.perf_counter()
            try:
                Res = RunSingle(Stack, Scen, Params)
                Wall = time_mod.perf_counter() - T0
                print(f"DONE {Wall:.1f}S | RMSE={Res['wind_rmse']:.3f}M/S | SEC={Res['sec_wh_km']:.2f}WH/KM")
                Cache[Key] = Res
                AllResults[Key] = Res
                SaveCache(Cache)
            except Exception as Ex:
                print(f"FAILED: {Ex}")
                import traceback
                traceback.print_exc()

print(f"\n{len(AllResults)}/{TotalRuns} RUNS COMPLETE")


# ─────────────────────────────────────────────────────────────────────────────
# BUILD DATAFRAME & COMPUTE METRICS
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
import math

ScenariosList = list(ScenarioDefs.keys())

def GetMetric(S, Sc, K):
    return AllResults.get(GetRunKey(S, Sc), {}).get(K, float("nan"))

for Scen in ScenariosList:
    Bl = GetMetric("baseline", Scen, "sec_wh_km")
    for Stack in StackNames:
        R = AllResults.get(GetRunKey(Stack, Scen))
        if R:
            R["energy_savings_pct"] = (Bl - R["sec_wh_km"]) / Bl * 100.0 if not math.isnan(Bl) and Bl > 0 else float("nan")

Rows = []
for Stack in StackNames:
    for Scen in ScenariosList:
        R = AllResults.get(GetRunKey(Stack, Scen))
        if not R:
            continue
        Rows.append({
            "Stack": StackLabels[Stack],
            "Scenario": Scen,
            "WindRMSE": R.get("wind_rmse", float("nan")),
            "WindSSRMSE": R.get("wind_ss_rmse", float("nan")),
            "ConvTime_ms": R.get("conv_time_ms", float("nan")),
            "TrkRMSE": R.get("tracking_rmse_m", float("nan")),
            "XTrkRMSE": R.get("cross_track_rmse_m", float("nan")),
            "MotorCurr": R.get("motor_current_as", float("nan")),
            "SEC": R.get("sec_wh_km", float("nan")),
            "SEC_aero": R.get("sec_aero_wh_km", float("nan")),
            "EnergySavPct": R.get("energy_savings_pct", float("nan")),
            "Endurance_s": R.get("endurance_s", float("nan")),
            "CombLat_ms": R.get("combined_lat_mean_ms", float("nan")),
            "CombLat_p95": R.get("combined_lat_p95_ms", float("nan")),
            "CombLat_max": R.get("combined_lat_max_ms", float("nan")),
            "ReplanLat": R.get("replan_lat_mean_ms", float("nan")),
            "GoalReached": float(R.get("goal_reached", False))
        })

Df = pd.DataFrame(Rows)
Mc = [C for C in Df.columns if C not in ("Stack", "Scenario")]
DfMean = Df.groupby("Stack", sort=False)[Mc].mean(numeric_only=True).reset_index()
DfMean["Scenario"] = "MEAN"

ShowCols = [
    ("Stack", "Stack"), ("WindRMSE", "WindRMSE(m/s)"), ("WindSSRMSE", "SS-RMSE"),
    ("ConvTime_ms", "Conv(ms)"), ("TrkRMSE", "WP-RMSE[*]"), ("XTrkRMSE", "XTrk-RMSE"),
    ("MotorCurr", "MotorCurr(As)"), ("SEC", "SEC(Wh/km)[A]"),
    ("EnergySavPct", "EnergySav%"), ("Endurance_s", "Endur(s)[A]"),
    ("CombLat_ms", "Lat_mean"), ("CombLat_max", "Lat_max"), ("GoalReached", "Reached")
]
Sk = [K for K, _ in ShowCols]
Rm = {K: V for K, V in ShowCols}
pd.set_option("display.float_format", lambda X: f"{X:.3f}")

print("TABLE 1 — PER-SCENARIO | [A]=ASSUMED BUS/BAT | [*]=NOT FAIR CROSS-REPLANNER")
for Scen in ScenariosList:
    Sub = Df[Df["Scenario"] == Scen][Sk].rename(columns=Rm)
    print(f"--- {Scen} ---")
    display(Sub.reset_index(drop=True))

print("\nTABLE 2 — MEAN OVER SCENARIOS")
display(DfMean[Sk].rename(columns=Rm).reset_index(drop=True))


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 3: DELTA VS BASELINE | TABLE 4: EKF CONTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────
print("TABLE 3 — DELTA VS BASELINE (MEAN; NEGATIVE = BETTER FOR ERROR/LATENCY)")
BlRow = DfMean[DfMean["Stack"] == StackLabels["baseline"]]
DeltaRows = []
for _, Row in DfMean.iterrows():
    if Row["Stack"] == StackLabels["baseline"]:
        continue
    D = {"Stack": Row["Stack"]}
    for Col in Mc:
        Bv = float(BlRow[Col].values[0]) if not BlRow.empty else float("nan")
        D[f"D_{Col}"] = float(Row[Col]) - Bv
    DeltaRows.append(D)
display(pd.DataFrame(DeltaRows).reset_index(drop=True))

print("\nTABLE 4 — EKF CONTRIBUTION (FULL - EKF_ABLATION; SAME ADAPTIVE+MPPI, ONLY EKF DIFFERS)")
Fr = DfMean[DfMean["Stack"] == StackLabels["full"]]
Ar = DfMean[DfMean["Stack"] == StackLabels["ekf_ablation"]]
EkfR = [
    {
        "Metric": C,
        "Full": float(Fr[C].values[0]),
        "EKF_Ablation": float(Ar[C].values[0]),
        "Delta": float(Fr[C].values[0]) - float(Ar[C].values[0])
    }
    for C in Mc if not Fr.empty and not Ar.empty
]
display(pd.DataFrame(EkfR).reset_index(drop=True))


# ─────────────────────────────────────────────────────────────────────────────
# DRAG MISMATCH VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────
import inspect
print("-- DRAG MISMATCH VERIFICATION --")
for L in inspect.getsource(ip.UAVPlantDynamics.derivative).split("\n"):
    if "KDrag" in L or "a_drag" in L:
        print(f"  Plant.derivative: {L.strip()}")
print(f"  -> PLANT COEFF = KDrag/Mass = 0.28/1.5 = {0.28/1.5:.5f}")

for L in inspect.getsource(ip.WindEKFV6.__init__).split("\n"):
    if "KDrag" in L:
        print(f"  WindEKFV6.__init__: {L.strip()}")
print(f"  -> EKF ASSUMES KDragFixed = {KDragNominal}")
print(f"  -> MISMATCH: EKF ASSUMES {KDragNominal/(0.28/1.5):.3f}X STRONGER DRAG THAN PLANT")

S1f = GetMetric("full", "S1_turbulent", "wind_rmse")
S1d = GetMetric("diagnostic", "S1_turbulent", "wind_rmse")
print(f"\n  S1 WindRMSE Full(0.28)={S1f:.3f}  Diag(0.28/1.5)={S1d:.3f}  delta={S1f-S1d:+.3f} m/s")


# ─────────────────────────────────────────────────────────────────────────────
# LATENCY FLAGS (CTRL/EKF < 20MS, REPLAN < 500MS)
# ─────────────────────────────────────────────────────────────────────────────
print("LATENCY FLAGS (WALL-CLOCK, NOT INJECTED INTO PLANT)")
for Key, R in sorted(AllResults.items()):
    Flags = []
    if R.get("ctrl_lat_max_ms", 0) > 20:
        Flags.append(f"CTRL>{R['ctrl_lat_max_ms']:.0f}ms")
    if R.get("ekf_lat_max_ms", 0) > 20:
        Flags.append(f"EKF>{R['ekf_lat_max_ms']:.0f}ms")
    if R.get("replan_lat_max_ms", 0) > 500:
        Flags.append(f"REPLAN>{R['replan_lat_max_ms']:.0f}ms")
    print(f"  {Key:<45} {'VIOLATIONS: ' + ', '.join(Flags) if Flags else 'OK'}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 9,
    "figure.dpi": 120
})

C = {
    "full": "#2ecc71",
    "baseline": "#e74c3c",
    "ekf_ablation": "#3498db",
    "diagnostic": "#f39c12"
}
Sh = {
    "full": "Full",
    "baseline": "Baseline",
    "ekf_ablation": "EKF-Abl",
    "diagnostic": "[DIAG]"
}

Mp = [
    ("WindRMSE", "Wind RMSE(m/s)", False),
    ("WindSSRMSE", "SS Wind RMSE", False),
    ("ConvTime_ms", "Conv Time(ms)", False),
    ("XTrkRMSE", "X-Track RMSE(m)", False),
    ("MotorCurr", "Motor Curr(As)[A]", False),
    ("SEC", "SEC(Wh/km)[A]", False),
    ("EnergySavPct", "Energy Sav%", True),
    ("Endurance_s", "Endurance(s)[A]", True)
]

Fig, Axes = plt.subplots(2, 4, figsize=(14, 6))
Axes = Axes.flatten()
X = np.arange(len(ScenariosList))
Bw = 0.18

for Ai, (Col, YLabel, Hib) in enumerate(Mp):
    Ax = Axes[Ai]
    for Si, Stack in enumerate(StackNames):
        Vals = [GetMetric(Stack, Sc, Col) for Sc in ScenariosList]
        Ax.bar(X + (Si - 2 + 0.5) * Bw, Vals, Bw, color=C[Stack], alpha=0.85, label=Sh[Stack])
    Ax.set_title(YLabel, pad=3)
    Ax.set_xticks(X)
    Ax.set_xticklabels(["S1", "S2", "S3"], fontsize=7)
    Ax.grid(axis="y", alpha=0.3)
    if Ai == 0:
        Ax.legend(fontsize=6)
    if Hib:
        Ax.set_facecolor("#f0fff4")

Fig.suptitle("Turbulent Stack Comparison — 8 Metrics | [A]=ASSUMED 14.8V,74Wh | [DIAG]=diagnostic", fontsize=8)
Fig.tight_layout()
plt.savefig("turbulence_comparison_8metrics.png", bbox_inches="tight", dpi=120)
plt.show()

Fig2, Ax2s = plt.subplots(1, 3, figsize=(14, 4))
for Ai, Scen in enumerate(ScenariosList):
    Ax = Ax2s[Ai]
    for Stack in StackNames:
        R = AllResults.get(GetRunKey(Stack, Scen), {})
        Wh = R.get("wind_hat_history", [])
        Wt = R.get("wind_true_history", [])
        N = min(len(Wh), len(Wt))
        if N == 0:
            continue
        E = [np.linalg.norm(np.asarray(Wh[I]) - np.asarray(Wt[I])) for I in range(N)]
        Ax.plot(np.arange(N) * Dt, E, color=C[Stack], lw=0.8, alpha=0.75, label=Sh[Stack])
    Ax.axvline(FlightSteps * Dt * 0.25, color="grey", ls="--", lw=0.6, alpha=0.5)
    Ax.set_title(f"{Scen}\nWind Error vs Time")
    Ax.set_xlabel("t(s)")
    Ax.set_ylabel("|w_hat-w|(m/s)")
    Ax.grid(alpha=0.3)
    Ax.legend(fontsize=6)

Fig2.suptitle("Wind Estimation Error vs Time")
Fig2.tight_layout()
plt.savefig("turbulence_comparison_wind_error.png", bbox_inches="tight", dpi=120)
plt.show()

print("PLOTS SAVED")


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT CSV
# ─────────────────────────────────────────────────────────────────────────────
import csv

CsvPath = "turbulence_stack_comparison.csv"
Fields = [
    "stack", "scenario", "wind_rmse", "wind_ss_rmse", "conv_time_ms",
    "tracking_rmse_m", "cross_track_rmse_m", "motor_current_as",
    "sec_wh_km", "sec_aero_wh_km", "energy_savings_pct", "endurance_s",
    "combined_lat_mean_ms", "combined_lat_p95_ms", "combined_lat_max_ms",
    "replan_lat_mean_ms", "replan_lat_max_ms",
    "goal_reached", "mission_time_s", "total_energy_j", "mean_power_w",
    "bus_voltage_assumed_V", "battery_wh_assumed"
]

with open(CsvPath, "w", newline="") as F:
    W = csv.DictWriter(F, fieldnames=Fields, extrasaction="ignore")
    W.writeheader()
    for Key, R in AllResults.items():
        Row = dict(R)
        Row.pop("wind_hat_history", None)
        Row.pop("wind_true_history", None)
        Row["bus_voltage_assumed_V"] = BusVoltageV
        Row["battery_wh_assumed"] = BatteryCapacityWh
        W.writerow(Row)

print(f"CSV EXPORTED: {CsvPath}")


# ─────────────────────────────────────────────────────────────────────────────
# CAVEATS
# ─────────────────────────────────────────────────────────────────────────────
print("""
CAVEATS
=======
1. COMPUTE LATENCY: WALL-CLOCK SIL TIME ONLY; NOT INJECTED INTO PLANT.
   EMBEDDED TIMING WILL DIFFER SIGNIFICANTLY.

2. ELECTRICAL METRICS [A]: BUS_VOLTAGE=14.8V, BATTERY=74WH ASSUMED
   (NOT FOUND IN REPO). ALL [A]-TAGGED TABLE CELLS DEPEND ON THESE.

3. WP-RMSE [*]: NOT FAIR CROSS-REPLANNER COMPARISON. MPPI TRAJECTORIES
   ARE 3-5 M/S VS GREEDY 2 M/S; WAYPOINT DISTANCE REFLECTS GEOMETRY,
   NOT PRECISION. USE X-TRACK RMSE FOR FAIR NAVIGATION ACCURACY.

4. DRAG MISMATCH: WindEKFV6 USES KDragFixed=0.28, PLANT USES 0.28/1.5.
   EKF ASSUMES 1.5X STRONGER DRAG THAN THE PLANT.
   DIAGNOSTIC STACK CORRECTS THIS (SEE CELL 11).

5. ADAPTIVE LTV MPC IGNORES WIND_EST (USES OWN DISTURBANCE OBSERVER).
   EKF ESTIMATE REACHES REPLANNERS ONLY, NOT DIRECTLY TO CONTROLLER.

6. SLSQP BASELINE: ~100-200MS/CALL -> ~200S/RUN. TOTAL ~10 MIN FIRST RUN.
   CACHED TO JSON; RE-RUNS ARE INSTANT.

7. SERIAL EXECUTION ENSURES LATENCY COMPARABILITY. PARALLELISING
   WOULD INVALIDATE TIMING MEASUREMENTS.
""")


# ─────────────────────────────────────────────────────────────────────────────
# FINDINGS (ALL NUMBERS FROM EXECUTED RESULTS)
# ─────────────────────────────────────────────────────────────────────────────
def MeanVal(Stack, Col):
    Row = DfMean[DfMean["Stack"] == StackLabels[Stack]]
    return float(Row[Col].values[0]) if not Row.empty else float("nan")

def FormatVal(V, D=3, S=""):
    return "N/A" if math.isnan(V) else f"{V:.{D}f}{S}"

print("=" * 70)
print("FINDINGS — ALL NUMBERS FROM EXECUTED SIMULATION RESULTS")
print("=" * 70)
print()
print("1. EKF WIND ESTIMATION")
for St in StackNames:
    print(f"   {StackLabels[St][:50]:<50} RMSE={FormatVal(MeanVal(St, 'WindRMSE'))} SS={FormatVal(MeanVal(St, 'WindSSRMSE'))} CONV={FormatVal(MeanVal(St, 'ConvTime_ms'), 0, 'ms')}")
Dw = MeanVal("full", "WindRMSE") - MeanVal("ekf_ablation", "WindRMSE")
print(f"   EKF CONTRIBUTION (FULL - EKF_ABLATION) DELTA={FormatVal(Dw)} M/S ({'CURRENT EKF IMPROVES' if Dw < 0 else 'CURRENT EKF DEGRADES'})")
Dd = MeanVal("full", "WindRMSE") - MeanVal("diagnostic", "WindRMSE")
print(f"   DRAG CORRECTION GAIN: DELTA={FormatVal(Dd)} M/S ({'CORRECTED-K BETTER' if Dd > 0 else 'CORRECTED-K WORSE OR SAME'})")
print()
print("2. ENERGY [ASSUMED BUS=14.8V, BAT=74WH]")
print(f"   SEC(Wh/km): FULL={FormatVal(MeanVal('full', 'SEC'))} BASELINE={FormatVal(MeanVal('baseline', 'SEC'))} EKF-ABL={FormatVal(MeanVal('ekf_ablation', 'SEC'))}")
print(f"   ENERGY SAVINGS FULL VS BASELINE: {FormatVal(MeanVal('full', 'EnergySavPct'), 1, '%')}")
print(f"   ENDURANCE: FULL={FormatVal(MeanVal('full', 'Endurance_s'), 0, 's')} BASELINE={FormatVal(MeanVal('baseline', 'Endurance_s'), 0, 's')}")
print()
print("3. NAVIGATION ACCURACY")
print(f"   X-TRACK RMSE: FULL={FormatVal(MeanVal('full', 'XTrkRMSE'))}M BASELINE={FormatVal(MeanVal('baseline', 'XTrkRMSE'))}M")
print(f"   WP-RMSE [* NOT COMPARABLE]: FULL={FormatVal(MeanVal('full', 'TrkRMSE'))}M BASELINE={FormatVal(MeanVal('baseline', 'TrkRMSE'))}M")
print()
print("4. LATENCY (WALL-CLOCK; NOT INJECTED INTO PLANT)")
print(f"   EKF+CTRL MEAN: FULL={FormatVal(MeanVal('full', 'CombLat_ms'))}MS BASELINE={FormatVal(MeanVal('baseline', 'CombLat_ms'))}MS")
print(f"   SLSQP REPLAN MEAN (BASELINE): {FormatVal(MeanVal('baseline', 'ReplanLat'))}MS")
print()
print("=" * 70)


