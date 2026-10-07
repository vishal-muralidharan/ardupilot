"""
UAV Plant Dynamics & Aerodynamics Environment Module.
Implements non-linear quadrotor planar state-space dynamics with quadratic aerodynamic drag
and configurable relative wind disturbance vectors using Runge-Kutta 4th Order (RK4) integration.
"""

from typing import Optional, Tuple, Union
import numpy as np


class UAVDynamics:
    """Non-linear 2D UAV planar model with quadratic aerodynamic drag and wind disturbance.

    State:
        x = [x, y, v_x, v_y]^T
    Control:
        u = [u_ax, u_ay]^T (acceleration commands in m/s^2)
    Aerodynamics:
        k_drag = 0.28
        Relative velocity v_rel = [v_x - w_x, v_y - w_y]^T
        F_drag / m = - k_drag * ||v_rel||_2 * v_rel
    """

    def __init__(
        self,
        k_drag: float = 0.28,
        dt: float = 0.1,
        u_bounds: Tuple[float, float] = (-5.0, 5.0),
    ):
        self.k_drag = float(k_drag)
        self.dt = float(dt)
        self.u_min, self.u_max = u_bounds

    def continuous_dynamics(
        self,
        x: np.ndarray,
        u: np.ndarray,
        wind: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Computes dx/dt = f(x, u, wind).

        Args:
            x: Current state vector [x, y, v_x, v_y] of shape (4,).
            u: Acceleration command [u_ax, u_ay] of shape (2,).
            wind: Wind velocity vector [w_x, w_y] of shape (2,). Default [0.0, 0.0].

        Returns:
            dx/dt state derivative of shape (4,).
        """
        x = np.asarray(x, dtype=np.float64)
        u = np.clip(np.asarray(u, dtype=np.float64), self.u_min, self.u_max)
        if wind is None:
            w = np.zeros(2, dtype=np.float64)
        else:
            w = np.asarray(wind, dtype=np.float64)

        vx = x[2]
        vy = x[3]
        v_rel_x = vx - w[0]
        v_rel_y = vy - w[1]
        v_rel_norm = np.sqrt(v_rel_x**2 + v_rel_y**2 + 1e-12)

        # Quadratic drag law: a_drag = - k_drag * ||v_rel|| * v_rel
        a_drag_x = -self.k_drag * v_rel_norm * v_rel_x
        a_drag_y = -self.k_drag * v_rel_norm * v_rel_y

        ax = u[0] + a_drag_x
        ay = u[1] + a_drag_y

        return np.array([vx, vy, ax, ay], dtype=np.float64)

    def step(
        self,
        x: np.ndarray,
        u: np.ndarray,
        wind: Optional[np.ndarray] = None,
        dt: Optional[float] = None,
    ) -> np.ndarray:
        """Advances state by dt using standard Runge-Kutta 4th Order (RK4).

        Args:
            x: State vector [x, y, v_x, v_y].
            u: Control vector [u_ax, u_ay].
            wind: Wind disturbance vector [w_x, w_y].
            dt: Integration timestep (defaults to self.dt = 0.1s).

        Returns:
            Next state vector x_{k+1}.
        """
        h = self.dt if dt is None else float(dt)
        x0 = np.asarray(x, dtype=np.float64)
        u0 = np.clip(np.asarray(u, dtype=np.float64), self.u_min, self.u_max)

        k1 = self.continuous_dynamics(x0, u0, wind)
        k2 = self.continuous_dynamics(x0 + 0.5 * h * k1, u0, wind)
        k3 = self.continuous_dynamics(x0 + 0.5 * h * k2, u0, wind)
        k4 = self.continuous_dynamics(x0 + h * k3, u0, wind)

        x_next = x0 + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return x_next

    def linearize(
        self,
        x_lin: np.ndarray,
        u_lin: np.ndarray,
        wind: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Analytical continuous-time linearization (A, B) around operating point (x_lin, u_lin).

        dx/dt ≈ A * delta_x + B * delta_u + f(x_lin, u_lin)
        """
        if wind is None:
            w = np.zeros(2, dtype=np.float64)
        else:
            w = np.asarray(wind, dtype=np.float64)

        vrx = x_lin[2] - w[0]
        vry = x_lin[3] - w[1]
        v_norm = np.sqrt(vrx**2 + vry**2 + 1e-12)

        # d(a_drag)/d(vx) = - k * [ v_norm + vrx^2 / v_norm ]
        # d(a_drag_x)/d(vy) = - k * [ vrx * vry / v_norm ]
        k = self.k_drag
        dax_dvx = -k * (v_norm + (vrx**2) / v_norm)
        dax_dvy = -k * (vrx * vry) / v_norm
        day_dvx = -k * (vrx * vry) / v_norm
        day_dvy = -k * (v_norm + (vry**2) / v_norm)

        A = np.zeros((4, 4), dtype=np.float64)
        A[0, 2] = 1.0
        A[1, 3] = 1.0
        A[2, 2] = dax_dvx
        A[2, 3] = dax_dvy
        A[3, 2] = day_dvx
        A[3, 3] = day_dvy

        B = np.zeros((4, 2), dtype=np.float64)
        B[2, 0] = 1.0
        B[3, 1] = 1.0

        return A, B

    def discrete_linearization(
        self,
        x_lin: np.ndarray,
        u_lin: np.ndarray,
        wind: Optional[np.ndarray] = None,
        dt: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """First-order discrete-time approximation (A_d, B_d) = (I + A*dt, B*dt)."""
        h = self.dt if dt is None else float(dt)
        A, B = self.linearize(x_lin, u_lin, wind)
        Ad = np.eye(4, dtype=np.float64) + A * h
        Bd = B * h
        return Ad, Bd
