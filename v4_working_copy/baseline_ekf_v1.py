"""
Telemetry-based Wind Estimation EKF (Module 1).

State vector (3,):
    x = [wx, wy, k_drag]
        wx, wy   : horizontal wind velocity components in the local NE frame (m/s)
        k_drag   : effective linear horizontal drag coefficient divided by mass,
                   i.e. units of 1/s. This is the "drag/gust correlation
                   parameter" referenced in the task brief -- it is what lets the
                   filter convert an unexplained (thrust-vs-motion) acceleration
                   residual into a wind estimate, and it is what gets tuned
                   offline against DataverseNO flight logs.

Process model (short-correlation gust model):
    Wind is modeled as a first-order Gauss-Markov ("Dryden-like") process:
        wx_dot = -wx / tau + noise
        wy_dot = -wy / tau + noise
    which, over a discrete step dt, gives the linear discrete transition
        wx[k+1] = wx[k] * (1 - dt/tau) + process_noise
    tau is the gust correlation time constant (seconds) -- short tau means the
    filter tracks fast-changing gusts more aggressively but is noisier; long tau
    smooths toward a slowly-varying mean wind. This is a hyperparameter, not a
    state, by design: making it part of the state would make the process model
    bilinear and considerably harder to defend/derive by hand.

    k_drag is modeled as a slow random walk (it should not change quickly in
    flight; it mostly captures airframe-to-airframe / configuration differences
    and is what DataverseNO-based offline tuning calibrates).

Measurement model (thrust-to-motion residual inversion):
    At each 50 Hz tick we are given, from standard autopilot telemetry only:
        v_ground : horizontal ground velocity, from GPS/IMU fusion (m/s, NE)
        a_thrust : horizontal specific force the autopilot's own thrust command
                   should have produced, computed from motor RPM/current and
                   attitude (m/s^2, NE) -- i.e. "commanded thrust" rotated into
                   the inertial frame and converted to acceleration
        a_meas   : horizontal specific force actually measured by the IMU
                   (m/s^2, NE)

    The residual y = a_meas - a_thrust is the acceleration the autopilot did NOT
    command and cannot otherwise explain. Under a linear drag assumption, this
    residual is modeled as aerodynamic drag against the *relative* airflow:
        y_pred = h(x) = -k_drag * (v_ground - w)
    where w = [wx, wy]. This is exactly "inverting thrust-to-motion residuals":
    we observe y and v_ground and solve for the wind vector w (and, over many
    samples, for k_drag) that best explains it.

This module has no autopilot-specific or dataset-specific code in it -- see
data_io.py for how DataverseNO-shaped logs get turned into (v_ground, a_thrust,
a_meas) triples that this class consumes.
"""

from __future__ import annotations

import numpy as np


class WindEKF:
    """A 3-state Extended Kalman Filter for horizontal wind + drag coefficient."""

    STATE_DIM = 3  # [wx, wy, k_drag]
    MEAS_DIM = 2   # [residual_x, residual_y]

    def __init__(
        self,
        tau_gust: float = 8.0,
        k_drag_init: float = 0.30,
        wind_init: tuple[float, float] = (0.0, 0.0),
        p0_wind: float = 4.0,
        p0_kdrag: float = 0.05,
        q_wind: float = 0.5,
        q_kdrag: float = 1e-5,
        r_meas: float = 0.02,
    ) -> None:
        """
        Parameters
        ----------
        tau_gust : gust correlation time constant (s) for the Gauss-Markov wind model.
        k_drag_init : initial guess for the drag/mass coefficient (1/s).
        wind_init : initial (wx, wy) guess (m/s), typically (0, 0).
        p0_wind, p0_kdrag : initial state covariance diagonal entries (variance).
        q_wind, q_kdrag : process noise variance per second for wind and k_drag.
        r_meas : measurement noise variance (m/s^2)^2 on each residual axis.

        NOTE ON DEFAULTS: q_wind and r_meas above are set to be consistent with
        the synthetic validation data's injected noise/gust statistics (see
        scripts/generate_synthetic_dataverseno.py), as a physically-reasonable
        starting point -- NOT values tuned against real DataverseNO logs. Real
        offline tuning against actual flight data (the next step in the task
        brief) will very likely want to revisit these, particularly r_meas
        (real IMU/attitude noise) and q_wind (real atmospheric turbulence
        intensity, which is airfield- and altitude-dependent).
        """
        self.tau = float(tau_gust)

        self.x = np.array([wind_init[0], wind_init[1], k_drag_init], dtype=float)
        self.P = np.diag([p0_wind, p0_wind, p0_kdrag]).astype(float)

        # Continuous-time process noise spectral densities; discretized in predict().
        self._q_wind = float(q_wind)
        self._q_kdrag = float(q_kdrag)
        self.R = np.diag([r_meas, r_meas]).astype(float)

    # ------------------------------------------------------------------ #
    # Prediction step
    # ------------------------------------------------------------------ #
    def predict(self, dt: float) -> None:
        """Advance the state and covariance by dt seconds (no measurement)."""
        decay = 1.0 - dt / self.tau
        F = np.array(
            [
                [decay, 0.0, 0.0],
                [0.0, decay, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        self.x = F @ self.x

        Q = np.diag(
            [
                self._q_wind * dt,
                self._q_wind * dt,
                self._q_kdrag * dt,
            ]
        )
        self.P = F @ self.P @ F.T + Q

    # ------------------------------------------------------------------ #
    # Measurement / update step
    # ------------------------------------------------------------------ #
    def _predict_measurement(self, v_ground: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (y_pred, H) for the current state and a given v_ground."""
        wx, wy, k = self.x
        vx, vy = v_ground

        y_pred = np.array(
            [
                -k * (vx - wx),
                -k * (vy - wy),
            ]
        )

        H = np.array(
            [
                [k, 0.0, -(vx - wx)],
                [0.0, k, -(vy - wy)],
            ]
        )
        return y_pred, H

    def update(self, v_ground: np.ndarray, a_thrust: np.ndarray, a_meas: np.ndarray) -> None:
        """
        Incorporate one measurement.

        v_ground : (2,) horizontal ground velocity [vx, vy] (m/s, NE)
        a_thrust : (2,) horizontal specific force expected from thrust command (m/s^2, NE)
        a_meas   : (2,) horizontal specific force measured by IMU (m/s^2, NE)
        """
        v_ground = np.asarray(v_ground, dtype=float)
        a_thrust = np.asarray(a_thrust, dtype=float)
        a_meas = np.asarray(a_meas, dtype=float)

        y_obs = a_meas - a_thrust
        y_pred, H = self._predict_measurement(v_ground)
        innovation = y_obs - y_pred

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ innovation
        I_KH = np.eye(self.STATE_DIM) - K @ H
        # Joseph form for numerical stability over long runs at 50 Hz.
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

    # ------------------------------------------------------------------ #
    # Convenience accessors
    # ------------------------------------------------------------------ #
    @property
    def wind(self) -> np.ndarray:
        """Current [wx, wy] wind estimate (m/s)."""
        return self.x[:2].copy()

    @property
    def k_drag(self) -> float:
        return float(self.x[2])

    @property
    def wind_speed(self) -> float:
        return float(np.linalg.norm(self.x[:2]))

    @property
    def wind_direction_deg(self) -> float:
        """Meteorological-style bearing the wind is blowing TOWARD, from North, deg."""
        wx, wy = self.x[:2]
        return float(np.degrees(np.arctan2(wx, wy)) % 360.0)

    def step(
        self,
        dt: float,
        v_ground: np.ndarray,
        a_thrust: np.ndarray,
        a_meas: np.ndarray,
    ) -> np.ndarray:
        """Convenience: predict then update in one call. Returns the wind estimate."""
        self.predict(dt)
        self.update(v_ground, a_thrust, a_meas)
        return self.wind
