"""Extended Kalman Filter (forward or backward) in augmented mean-element space.

State: [a, h, k, i, Omega, M, w_R, w_S, w_W]^T
(h, k) = (e*sin(argp), e*cos(argp))

Measurement is taken in (h, k) space so the Jacobian stays linear and
singularity-free: z = [a, h, k, i, Omega, M] = Hx with H = [I_6, 0].
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
    """Classical [a, e, i, Omega, argp, M] -> [a, h, k, i, Omega, M]."""
    z_hk = z.copy()
    e = z[1]
    argp = z[4]
    z_hk[1] = e * math.sin(argp)
    z_hk[2] = e * math.cos(argp)
    return z_hk


def _measurement_function(x: np.ndarray) -> np.ndarray:
    """Predicted measurement [a, h, k, i, Omega, M] from the 9-state."""
    return np.array([x[0], x[1], x[2], x[3], x[4], x[5]])


def _measurement_jacobian(x: np.ndarray, eps: float = 1.0e-7) -> np.ndarray:
    """H = [I_6, 0]. eps kept for API compatibility, unused."""
    H = np.zeros((6, 9))
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    H[2, 2] = 1.0
    H[3, 3] = 1.0
    H[4, 4] = 1.0
    H[5, 5] = 1.0
    return H


def _wrap_angle(a: float) -> float:
    """Wrap angle difference to (-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


class EKF:
    """9-state EKF with FOGM dynamic model compensation."""

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

    def predict(self, dt: float, f107: float, kp: float) -> None:
        # Sub-step to keep B*dt coupling from exploding the covariance.
        n_sub = max(1, int(abs(dt) / 3600.0))
        dt_sub = dt / n_sub

        for _ in range(n_sub):
            dxdt = augmented_dynamics(
                self.x, self.Cd, self.area, self.mass,
                self.epoch_jd, self.t, f107, kp, self.tau
            )

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

        z_hk = _classical_to_hk_measurement(z)

        el = _elements_from_x(self.x)
        e_est = el.ecc
        argp_est = el.argp

        # R_diag is classical [R_a, R_e, R_i, R_Omega, R_argp, R_M];
        # transform the e/argp block into h/k.
        R_hk_diag = np.array([
            R_diag[0],
            (math.sin(argp_est)**2 * R_diag[1] +
             (e_est * math.cos(argp_est))**2 * R_diag[4]),
            (math.cos(argp_est)**2 * R_diag[1] +
             (e_est * math.sin(argp_est))**2 * R_diag[4]),
            R_diag[2],
            R_diag[3],
            R_diag[5],
        ])
        R = np.diag(R_hk_diag)

        z_pred = _measurement_function(self.x)
        H = _measurement_jacobian(self.x)

        y = z_hk - z_pred
        y[4] = _wrap_angle(y[4])  # Omega
        y[5] = _wrap_angle(y[5])  # M

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

    def get_elements(self) -> MeanElements:
        return _elements_from_x(self.x)

    def get_accel(self) -> np.ndarray:
        return self.x[6:9].copy()

    def get_cov(self) -> np.ndarray:
        return self.P.copy()

    def record(self, label: str = "") -> None:
        self.history.append({
            "t": self.t,
            "x": self.x.copy(),
            "P": self.P.copy(),
            "label": label,
        })