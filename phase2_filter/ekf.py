"""Extended Kalman Filter (forward or backward) in augmented mean-element space.

REQUIRED COMPANION CHANGE to dynamics.py's (h,k) singularity fix: the state
is no longer [a, e, i, Omega, argp, M, w_R, w_S, w_W] - it's now
[a, h, k, i, Omega, M, w_R, w_S, w_W]. The old measurement update assumed
z = H @ x with H = [I_6, 0], i.e. that the first 6 states directly equal
the measured classical elements [a, e, i, Omega, argp, M]. That is no longer
true (x[1]=h != e, x[2]=k != i, x[3]=i != Omega, x[4]=Omega != argp - the
positions don't even line up, let alone the values), so the update must
become a proper nonlinear measurement function h(x) with a linearized
Jacobian H = dh/dx, computed via finite differences (consistent with how
compute_stm already handles the process model's Jacobian).
"""

import math
import numpy as np
from typing import Tuple, Optional, List, Dict

from propagator.orbital import MeanElements
from . import config
from .dynamics import (
    augmented_dynamics, compute_stm, compute_process_noise,
    _elements_from_x, e_argp_from_hk
)


def _measurement_function(x: np.ndarray) -> np.ndarray:
    """Predicted measurement z_pred = [a, e, i, Omega, argp, M] from the
    9-state vector [a, h, k, i, Omega, M, w_R, w_S, w_W]."""
    el = _elements_from_x(x)
    return np.array([el.a, el.ecc, el.inc, el.raan, el.argp, el.M])


def _measurement_jacobian(x: np.ndarray, eps: float = 1.0e-7) -> np.ndarray:
    """H = d(measurement)/dx via forward finite differences - same approach
    already used for the process model's STM, applied here for consistency
    rather than hand-deriving analytic partials through the atan2/hypot
    conversion (which has its own removable-but-fiddly behaviour right at
    h=k=0)."""
    z0 = _measurement_function(x)
    H = np.zeros((6, 9))
    for j in range(9):
        h = max(abs(x[j]), 1.0e-6) * eps
        x_plus = x.copy()
        x_plus[j] += h
        z_plus = _measurement_function(x_plus)
        H[:, j] = (z_plus - z0) / h
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
        """Propagate state and covariance over ``dt``.

        Forward filter : dt > 0,  P = Phi P Phi^T + S
        Backward filter: dt < 0,  P = Phi P Phi^T - S  (Bennett Sec. II.E)

        Unchanged from before - state-representation-agnostic.
        """
        dxdt = augmented_dynamics(
            self.x, self.Cd, self.area, self.mass,
            self.epoch_jd, self.t, f107, kp, self.tau
        )
        self.x += dxdt * dt

        Phi = compute_stm(
            self.x, dt, self.Cd, self.area, self.mass,
            self.epoch_jd, self.t, f107, kp, self.tau
        )

        S = compute_process_noise(dt, self.tau, self.q)

        self.P = Phi @ self.P @ Phi.T
        if self.direction == "forward":
            self.P += S
        else:
            self.P -= S
            eigvals, eigvecs = np.linalg.eigh(self.P)
            eigvals = np.maximum(eigvals, 1.0e-12)
            self.P = eigvecs @ np.diag(eigvals) @ eigvecs.T

        self.t += dt

    def update(self, z: np.ndarray, R_diag: np.ndarray) -> None:
        """Kalman update with measurement z = [a, e, i, Omega, argp, M].

        CHANGED: now uses the nonlinear measurement function and its
        finite-difference Jacobian (see module docstring), instead of the
        old linear H = [I_6, 0], which silently assumed the state directly
        equalled the measurement - no longer true after the (h,k) fix.

        Also wraps the argp and M innovations to (-pi, pi], since a true
        measured angle near +-pi could otherwise produce a spurious large
        innovation against a predicted angle just across the wrap boundary.
        """
        z = np.asarray(z, dtype=float)
        R = np.diag(R_diag)

        z_pred = _measurement_function(self.x)
        H = _measurement_jacobian(self.x)

        y = z - z_pred
        y[4] = _wrap_angle(y[4])   # argp
        y[5] = _wrap_angle(y[5])   # M

        S_cov = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S_cov)

        self.x += K @ y
        self.P = (np.eye(9) - K @ H) @ self.P

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