import numpy as np
from replanner_common import ReplannerBase

class Strategy3MPPI(ReplannerBase):
    """Energy-Aware MPPI Replanner Strategy."""

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.NumRollouts = self.Config.get("NumRollouts", 256)
        self.Temperature = self.Config.get("Temperature", 10.0)
        self.NoiseSigma = self.Config.get("NoiseSigma", 1.5)
        self.TrackWeight = self.Config.get("TrackWeight", 1.0)
        self.EnergyWeight = self.Config.get("EnergyWeight", 0.5)
        self.SmoothWeight = self.Config.get("SmoothWeight", 0.1)
        self.TerminalWeight = self.Config.get("TerminalWeight", 10.0)
        
        # Nominal control sequence (U) shape: (Horizon, 2)
        self.U = np.zeros((self.Horizon, 2))

    @property
    def name(self) -> str:
        return "EnergyAwareMPPI"

    def replan(self, state: np.ndarray, goal: np.ndarray, wind: np.ndarray) -> np.ndarray:
        # a. Generate K perturbed control sequences
        epsilon = np.random.normal(0, self.NoiseSigma, (self.NumRollouts, self.Horizon, 2))
        U_k = self.U + epsilon

        # b. Vectorized rollout through dynamics
        trajectories = np.zeros((self.NumRollouts, self.Horizon + 1, 4))
        trajectories[:, 0, :] = state

        for t in range(self.Horizon):
            controls = np.clip(U_k[:, t, :], -self.Dynamics.UMax, self.Dynamics.UMax)
            
            X = trajectories[:, t, 0]
            Y = trajectories[:, t, 1]
            Vx = trajectories[:, t, 2]
            Vy = trajectories[:, t, 3]
            Ax = controls[:, 0]
            Ay = controls[:, 1]

            VairX = Vx - wind[0]
            VairY = Vy - wind[1]
            AirSpeed = np.sqrt(VairX**2 + VairY**2) + 1e-8

            DragX = -self.Dynamics.KDrag * AirSpeed * VairX
            DragY = -self.Dynamics.KDrag * AirSpeed * VairY

            VxNew = Vx + (Ax + DragX) * self.Dt
            VyNew = Vy + (Ay + DragY) * self.Dt
            XNew = X + Vx * self.Dt
            YNew = Y + Vy * self.Dt

            trajectories[:, t + 1, 0] = XNew
            trajectories[:, t + 1, 1] = YNew
            trajectories[:, t + 1, 2] = VxNew
            trajectories[:, t + 1, 3] = VyNew

        # c. Compute cost for each rollout
        costs = np.zeros(self.NumRollouts)
        for t in range(self.Horizon):
            pos = trajectories[:, t, :2]
            dist_sq = np.sum((pos - goal)**2, axis=1)
            
            vx = trajectories[:, t, 2]
            vy = trajectories[:, t, 3]
            vair_x = vx - wind[0]
            vair_y = vy - wind[1]
            airspeed = np.sqrt(vair_x**2 + vair_y**2)
            power = self.Dynamics.KDrag * airspeed**3
            
            du = U_k[:, t, :] if t == 0 else (U_k[:, t, :] - U_k[:, t-1, :])
            smooth_cost = np.sum(du**2, axis=1)
            
            costs += self.TrackWeight * dist_sq + self.EnergyWeight * power * self.Dt + self.SmoothWeight * smooth_cost
            
        term_pos = trajectories[:, self.Horizon, :2]
        term_dist_sq = np.sum((term_pos - goal)**2, axis=1)
        costs += self.TerminalWeight * term_dist_sq

        # d. Compute weights via softmin
        beta = np.min(costs)
        exp_weights = np.exp(-1.0 / self.Temperature * (costs - beta))
        weights = exp_weights / np.sum(exp_weights)

        # e. Update nominal controls
        self.U = np.sum(weights[:, np.newaxis, np.newaxis] * U_k, axis=0)

        # 3. Roll out the updated U to get the reference trajectory
        ref_traj_full = self.Dynamics.rollout(state, self.U, wind, self.Dt)
        ref_traj = ref_traj_full[1:]  # (Horizon, 4)

        # 4. Shift U left by one step for warm-starting next call
        self.U[:-1] = self.U[1:]
        self.U[-1] = np.zeros(2)

        return ref_traj

if __name__ == "__main__":
    mppi = Strategy3MPPI()
    state = np.array([0.0, 0.0, 0.0, 0.0])
    goal = np.array([10.0, 10.0])
    wind = np.array([2.0, 0.0])
    
    print("Testing MPPI Replanner...")
    traj = mppi.replan(state, goal, wind)
    print(f"Output shape: {traj.shape}, Expected: (30, 4)")
    print("Smoke test passed.")
