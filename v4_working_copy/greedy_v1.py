"""
Strategy 6: Greedy Wind-Vector Field Following.
"""
import numpy as np
from replanner_common import ReplannerBase

class Strategy6Greedy(ReplannerBase):
    """
    Greedy Wind-Vector Field Following Replanner.
    """
    def __init__(self, Config: dict = None):
        super().__init__(Config)
        self.BlendWeight = self.Config.get("BlendWeight", 0.3)
        self.MaxDeviation = self.Config.get("MaxDeviation", 10.0)

    @property
    def name(self) -> str:
        return "Strategy6_Greedy"

    def replan(self, State: np.ndarray, Goal: np.ndarray, Wind: np.ndarray) -> np.ndarray:
        Pos = State[:2]
        Dir = Goal - Pos
        Dist = np.linalg.norm(Dir)

        if Dist < 1e-6:
            return self._make_straight_line_ref(State, Goal)

        # 1. Compute nominal heading toward goal
        ThetaGoal = np.arctan2(Dir[1], Dir[0])

        # 2. Compute wind-optimal heading
        Angles = np.linspace(0, 2*np.pi, 360, endpoint=False)
        Costs = np.zeros_like(Angles)
        for i, Theta in enumerate(Angles):
            DirVec = np.array([np.cos(Theta), np.sin(Theta)])
            Costs[i] = self.EnergyModel.power(self.VCruise * DirVec, Wind)
        ThetaWind = Angles[np.argmin(Costs)]

        # 3. Blend, with max lateral deviation limit
        # Handle angular wrap-around for interpolation
        DeltaTheta = (ThetaWind - ThetaGoal + np.pi) % (2 * np.pi) - np.pi
        
        # 4. Limit deviation
        Beta = self.BlendWeight
        BetaMax = min(Beta, self.MaxDeviation / Dist) if Dist > 0 else 0.0
        
        ThetaRef = ThetaGoal + BetaMax * DeltaTheta

        # 5. Generate straight-line reference trajectory at θ_ref
        RefTraj = np.zeros((self.Horizon, 4))
        DirUnit = np.array([np.cos(ThetaRef), np.sin(ThetaRef)])
        Vel = DirUnit * self.VCruise

        # Ensure we don't overshoot goal in distance
        for t in range(self.Horizon):
            Progress = (t + 1) * self.Dt * self.VCruise
            if Progress < Dist:
                RefTraj[t, :2] = Pos + DirUnit * Progress
                RefTraj[t, 2:4] = Vel
            else:
                RefTraj[t, :2] = Goal
                RefTraj[t, 2:4] = np.zeros(2)

        return RefTraj

if __name__ == "__main__":
    Strat = Strategy6Greedy()
    State = np.array([0.0, 0.0, 0.0, 0.0])
    Goal = np.array([50.0, 0.0])
    Wind = np.array([-3.0, 0.0])
    Ref = Strat.replan(State, Goal, Wind)
    print("Strategy 6 Greedy - First 5 reference points (Headwind):")
    print(Ref[:5])
