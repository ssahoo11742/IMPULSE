"""Extended Kalman Filter (forward or backward) in augmented mean-element space.

State: [a, h, k, i, Omega, M, w_R, w_S, w_W]^T
(h, k) = (e*sin(argp), e*cos(argp))

MEASUREMENT FIX: Instead of converting the state back to classical (e, argp)
for the measurement (which reintroduces the 1/e singularity in the Jacobian),
we transform the incoming classical measurement into (h, k) space.
The measurement model is therefore linear: z = [a, h, k, i, Omega, M] = Hx
with H = [I_6, 0].
"""

import math
import numpy as np
from typing import Tuple, Optional, List, Dict

from constants.constants import R_EARTH
from propagator.orbital import MeanElements
from . import config
from .linalg_utils import safe_eigh_floor
from .dynamics import (
    augmented_dynamics, compute_stm, compute_process_noise,
    _elements_from_x, e_argp_from_hk
)


def _classical_to_hk_measurement(z: np.ndarray) -> np.ndarray:
    """Transform classical element measurement [a, e, i, Omega, argp, M]
    to non-singular [a, h, k, i, Omega, M]."""
    z_hk = z.copy()
    e = z[1]
    argp = z[4]
    z_hk[1] = e * math.sin(argp)
    z_hk[2] = e * math.cos(argp)
    return z_hk


def _measurement_function(x: np.ndarray) -> np.ndarray:
    """Predicted measurement z_pred = [a, h, k, i, Omega, M] from the
    9-state vector [a, h, k, i, Omega, M, w_R, w_S, w_W].

    Direct readout — no conversion back to (e, argp), no singularity.
    """
    return np.array([x[0], x[1], x[2], x[3], x[4], x[5]])


def _measurement_jacobian(x: np.ndarray, eps: float = 1.0e-7) -> np.ndarray:
    """H = d(measurement)/dx for measurement [a, h, k, i, Omega, M].

    Since the measurement is just the first 6 states directly, H = [I_6, 0].
    The eps parameter is kept for API compatibility but is no longer used.
    """
    H = np.zeros((6, 9))
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    H[2, 2] = 1.0
    H[3, 3] = 1.0
    H[4, 4] = 1.0
    H[5, 5] = 1.0
    return H


def _wrap_angle(a: float) -> float:
    """Wrap an angle difference to (-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


class EKF:
    """9-state EKF with FOGM dynamic model compensation.

    State: [a, h, k, i, Omega, M, w_R, w_S, w_W]^T
    (h, k) = (e*sin(argp), e*cos(argp)) - see dynamics.py for why.
    """

    def __init__(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
        Cd: float,
        area: float,
        mass: float,
        epoch_jd: float,
        tau: float,
        q: float,
        direction: str = "forward",
    ):
        assert direction in ("forward", "backward")
        self.x = x0.copy().astype(float)
        self.P = P0.copy().astype(float)
        self.Cd = Cd
        self.area = area
        self.mass = mass
        self.epoch_jd = epoch_jd
        self.tau = tau if direction == "forward" else -tau
        self.q = q
        self.direction = direction
        self.t = 0.0  # elapsed seconds from epoch_jd

        self.history: List[Dict] = []

    # ------------------------------------------------------------------
    # Core filter steps
    # ------------------------------------------------------------------

    def predict(self, dt: float, f107: float, kp: float) -> None:
        # Sub-step to prevent covariance explosion from large B*dt coupling.
        # 1-hour sub-steps are a good compromise between accuracy and speed.
        n_sub = max(1, int(abs(dt) / 3600.0))
        dt_sub = dt / n_sub
        
        for _ in range(n_sub):
            dxdt = augmented_dynamics(
                self.x, self.Cd, self.area, self.mass,
                self.epoch_jd, self.t, f107, kp, self.tau
            )
            
            # Defensive: if this sub-step would drive a negative, coast
            if not np.all(np.isfinite(dxdt)) or (self.x[0] + dxdt[0] * dt_sub) <= 0:
                self.t += dt_sub
                continue
                
            self.x += dxdt * dt_sub

            Phi = compute_stm(
                self.x, dt_sub, self.Cd, self.area, self.mass,
                self.epoch_jd, self.t, f107, kp, self.tau
            )

            S = compute_process_noise(dt_sub, self.tau, self.q)

            self.P = Phi @ self.P @ Phi.T
                    # Numerical divergence guard
            if not np.all(np.isfinite(self.P)) or np.any(np.diag(self.P) > 1e12):
                raise RuntimeError("EKF covariance diverged during predict")
            if self.direction == "forward":
                self.P += S
            else:
                self.P -= S
                self.P = safe_eigh_floor(self.P)

            self.t += dt_sub

    def update(self, z: np.ndarray, R_diag: np.ndarray) -> None:
        z = np.asarray(z, dtype=float)

        # Transform classical measurement to (h, k) space
        z_hk = _classical_to_hk_measurement(z)

        # Get e, argp from current state estimate for R transformation
        el = _elements_from_x(self.x)
        e_est = el.ecc
        argp_est = el.argp

        # Transform measurement noise covariance from classical to (h, k).
        # R_diag = [R_a, R_e, R_i, R_Omega, R_argp, R_M] (variances).
        R_hk_diag = np.array([
            R_diag[0],                                    # a
            (math.sin(argp_est)**2 * R_diag[1] +
             (e_est * math.cos(argp_est))**2 * R_diag[4]),  # h
            (math.cos(argp_est)**2 * R_diag[1] +
             (e_est * math.sin(argp_est))**2 * R_diag[4]),  # k
            R_diag[2],                                    # i
            R_diag[3],                                    # Omega
            R_diag[5],                                    # M
        ])
        R = np.diag(R_hk_diag)

        z_pred = _measurement_function(self.x)
        H = _measurement_jacobian(self.x)

        y = z_hk - z_pred
        # Wrap angle differences: Omega (index 4) and M (index 5)
        y[4] = _wrap_angle(y[4])
        y[5] = _wrap_angle(y[5])

        # Defensive: if covariance or Jacobian is corrupted, skip this update
        if not np.all(np.isfinite(self.P)) or not np.all(np.isfinite(H)):
            return

        S_cov = H @ self.P @ H.T + R
        S_cov += np.eye(6) * 1.0e-12

        try:
            K = self.P @ H.T @ np.linalg.inv(S_cov)
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.pinv(S_cov)

        self.x += K @ y
        I_KH = np.eye(9) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_elements(self) -> MeanElements:
        """Return current mean elements (converted from the first 6 states)."""
        return _elements_from_x(self.x)

    def get_accel(self) -> np.ndarray:
        """Return current RSW unmodeled acceleration [w_R, w_S, w_W]."""
        return self.x[6:9].copy()

    def get_cov(self) -> np.ndarray:
        """Return current 9x9 covariance."""
        return self.P.copy()

    def record(self, label: str = "") -> None:
        """Append current state/covariance/time to history."""
        self.history.append({
            "t": self.t,
            "x": self.x.copy(),
            "P": self.P.copy(),
            "label": label,
        })