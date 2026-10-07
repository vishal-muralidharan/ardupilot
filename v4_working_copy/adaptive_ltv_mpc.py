"""
Adaptive LTV Disturbance-Observer MPC Controller.
Sixth paradigm completing the algorithmic suite. Combines Linear Time-Varying (LTV)
predictive control with a high-gain disturbance observer estimating effective wind
forces and aerodynamic drag on-the-fly for feedforward disturbance cancellation.
"""

from typing import Optional
import numpy as np
from scipy.optimize import minimize

from base import BaseController
from dynamics import UAVDynamics


class AdaptiveLTVDisturbanceObserverMPC(BaseController):
    """Adaptive LTV MPC with online disturbance observer (DOB)."""

    def __init__(
        self,
        dynamics: UAVDynamics,
        horizon: int = 16,
        q_pos: float = 12.0,
        q_vel: float = 2.5,
        r_ctrl: float = 0.08,
        observer_alpha: float = 0.35,
        max_iter: int = 25,
    ):
        super().__init__(name="Adaptive LTV Disturbance-Observer MPC")
        self.dynamics = dynamics
        self.N = horizon
        self.q_pos = q_pos
        self.q_vel = q_vel
        self.r_ctrl = r_ctrl
        self.observer_alpha = observer_alpha
        self.max_iter = max_iter

        self.u_prev = np.zeros(2 * self.N, dtype=np.float64)
        self.bounds = [(self.dynamics.u_min, self.dynamics.u_max) for _ in range(2 * self.N)]

        # Disturbance observer memory
        self.d_hat = np.zeros(2, dtype=np.float64)
        self.prev_v: Optional[np.ndarray] = None
        self.prev_u: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.u_prev = np.zeros(2 * self.N, dtype=np.float64)
        self.d_hat = np.zeros(2, dtype=np.float64)
        self.prev_v = None
        self.prev_u = None

    def _update_observer(self, current_v: np.ndarray, dt: float):
        """Updates estimated disturbance force d_hat = a_actual - u_applied."""
        if self.prev_v is not None and self.prev_u is not None:
            a_actual = (current_v - self.prev_v) / max(dt, 1e-4)
            d_instant = a_actual - self.prev_u
            # Exponential smoothing observer
            self.d_hat = (1.0 - self.observer_alpha) * self.d_hat + self.observer_alpha * d_instant
        self.prev_v = current_v.copy()

    def _cost(
        self,
        u_flat: np.ndarray,
        x0: np.ndarray,
        target: np.ndarray,
        dt: float,
    ) -> float:
        u_seq = u_flat.reshape(self.N, 2)
        total_cost = 0.0
        x_curr = x0.copy()

        for k in range(self.N):
            uk = u_seq[k]
            # Incorporate estimated lumped disturbance into LTV prediction
            # dot(v) = u + d_hat
            x_next = np.empty(4, dtype=np.float64)
            x_next[0] = x_curr[0] + x_curr[2] * dt
            x_next[1] = x_curr[1] + x_curr[3] * dt
            x_next[2] = x_curr[2] + (uk[0] + self.d_hat[0]) * dt
            x_next[3] = x_curr[3] + (uk[1] + self.d_hat[1]) * dt

            dx = x_next[0] - target[0]
            dy = x_next[1] - target[1]
            dvx = x_next[2] - target[2]
            dvy = x_next[3] - target[3]

            mult = 5.0 if k == self.N - 1 else 1.0
            pos_cost = mult * self.q_pos * (dx**2 + dy**2)
            vel_cost = mult * self.q_vel * (dvx**2 + dvy**2)
            ctrl_cost = self.r_ctrl * (uk[0] ** 2 + uk[1] ** 2)

            total_cost += pos_cost + vel_cost + ctrl_cost
            x_curr = x_next

        return float(total_cost)

    def compute_control(
        self,
        state: np.ndarray,
        target: np.ndarray,
        wind_magnitude: float,
        wind_vector: Optional[np.ndarray] = None,
        dt: float = 0.1,
    ) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        # 1. Update disturbance observer
        curr_v = state[2:4]
        self._update_observer(curr_v, dt)

        # 2. Warm start
        u_init = np.empty_like(self.u_prev)
        u_init[:-2] = self.u_prev[2:]
        u_init[-2:] = self.u_prev[-2:]

        # 3. Solve LTV-MPC with observer offset
        res = minimize(
            fun=self._cost,
            x0=u_init,
            args=(state, target, dt),
            method="SLSQP",
            bounds=self.bounds,
            options={
                "maxiter": self.max_iter,
                "ftol": 1e-4,
                "disp": False,
            },
        )

        if res.x is not None:
            self.u_prev = res.x
            u_cmd = res.x[:2]
        else:
            u_cmd = u_init[:2]

        u_cmd = np.clip(u_cmd, self.dynamics.u_min, self.dynamics.u_max)
        self.prev_u = u_cmd.copy()
        return u_cmd
