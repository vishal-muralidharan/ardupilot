"""
Single-Shooting SLSQP Baseline Controller.
Flattens the control sequence over horizon N=20, optimizing tracking error
and control effort directly via forward plant rollouts.
"""

from typing import Optional
import numpy as np
from scipy.optimize import minimize

from base import BaseController
from dynamics import UAVDynamics


class SingleShootingSLSQP(BaseController):
    """Single-shooting NMPC baseline using scipy.optimize.minimize (SLSQP)."""

    def __init__(
        self,
        dynamics: UAVDynamics,
        horizon: int = 20,
        q_pos: float = 12.0,
        q_vel: float = 2.5,
        r_ctrl: float = 0.08,
        q_term_mult: float = 5.0,
        max_iter: int = 20,
    ):
        super().__init__(name="Single-Shooting SLSQP (Baseline)")
        self.dynamics = dynamics
        self.N = horizon
        self.q_pos = q_pos
        self.q_vel = q_vel
        self.r_ctrl = r_ctrl
        self.q_term_mult = q_term_mult
        self.max_iter = max_iter

        # Decision variable: U of shape (2 * N,)
        self.u_prev = np.zeros(2 * self.N, dtype=np.float64)
        self.bounds = [
            (self.dynamics.u_min, self.dynamics.u_max) for _ in range(2 * self.N)
        ]

    def reset(self) -> None:
        self.u_prev = np.zeros(2 * self.N, dtype=np.float64)

    def _cost(
        self,
        u_flat: np.ndarray,
        x0: np.ndarray,
        target: np.ndarray,
        wind: Optional[np.ndarray],
        dt: float,
    ) -> float:
        u_seq = u_flat.reshape(self.N, 2)
        total_cost = 0.0
        x_curr = x0.copy()

        w_pos = self.q_pos
        w_vel = self.q_vel
        w_u = self.r_ctrl

        for k in range(self.N):
            u_k = u_seq[k]
            x_next = self.dynamics.step(x_curr, u_k, wind=wind, dt=dt)

            pos_err_sq = (x_next[0] - target[0]) ** 2 + (x_next[1] - target[1]) ** 2
            vel_err_sq = (x_next[2] - target[2]) ** 2 + (x_next[3] - target[3]) ** 2
            u_sq = u_k[0] ** 2 + u_k[1] ** 2

            mult = self.q_term_mult if k == self.N - 1 else 1.0
            total_cost += mult * (w_pos * pos_err_sq + w_vel * vel_err_sq) + w_u * u_sq
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
        # Warm start by shifting control sequence left
        u_init = np.empty_like(self.u_prev)
        u_init[:-2] = self.u_prev[2:]
        u_init[-2:] = self.u_prev[-2:]

        # Nominal wind assumption (open-loop baseline or estimated)
        w_est = wind_vector if wind_vector is not None else np.zeros(2)

        res = minimize(
            fun=self._cost,
            x0=u_init,
            args=(state, target, w_est, dt),
            method="SLSQP",
            bounds=self.bounds,
            options={
                "maxiter": self.max_iter,
                "ftol": 1e-4,
                "disp": False,
            },
        )

        if res.success or res.x is not None:
            self.u_prev = res.x
            u_opt = res.x[:2]
        else:
            u_opt = u_init[:2]

        return np.clip(u_opt, self.dynamics.u_min, self.dynamics.u_max)
