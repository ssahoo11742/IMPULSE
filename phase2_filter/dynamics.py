"""9-state augmented dynamics and STM for the debris-detection EKF.

State: x = [a, h, k, i, Omega, M, w_R, w_S, w_W]^T
where h = e*sin(argp), k = e*cos(argp) (non-singular eccentricity pair).

Units: metres, radians, seconds.
"""

import math
import numpy as np
from typing import Tuple

from propagator.orbital import (
    MeanElements, brouwer_rates, drag_rates, srp_ecc_rate,
    third_body_rates, mean_to_true_anomaly
)
from propagator.atmosphere import density
from propagator.ephemeris import sun_position_eci, moon_position_eci
from constants.constants import MU, R_EARTH

from . import config

# Floor for the remaining singular dM/dt term (see compute_B_matrix).
E_FLOOR_FOR_DM = 1.0e-8


def hk_from_e_argp(e: float, argp: float) -> Tuple[float, float]:
    """(e, argp) -> (h, k)."""
    return e * math.sin(argp), e * math.cos(argp)


def e_argp_from_hk(h: float, k: float) -> Tuple[float, float]:
    """(h, k) -> (e, argp). argp is arbitrary at e=0; return 0 by convention."""
    e = math.hypot(h, k)
    argp = math.atan2(h, k) if e > 1e-12 else 0.0
    return e, argp


def _elements_from_x(x: np.ndarray) -> MeanElements:
    e, argp = e_argp_from_hk(float(x[1]), float(x[2]))
    e = min(e, 0.999999)
    return MeanElements(
        a=float(x[0]),
        ecc=e,
        inc=float(x[3]),
        raan=float(x[4]),
        argp=argp,
        M=float(x[5])
    )


def compute_B_matrix(el: MeanElements) -> np.ndarray:
    """6x3 Gauss-VOP matrix: RSW accel -> [da, dh, dk, di, dOmega, dM]/dt.

    dh/dk rows use the chain-rule form so the 1/e singularity cancels.
    dM/dt still uses an explicit e floor (E_FLOOR_FOR_DM).
    """
    nu, _ = mean_to_true_anomaly(el.M, el.ecc)
    a, e, i, argp = el.a, el.ecc, el.inc, el.argp

    p = a * (1.0 - e * e)
    r = p / (1.0 + e * math.cos(nu))
    h_ang = math.sqrt(MU * p)
    b_semi = a * math.sqrt(max(0.0, 1.0 - e * e))

    cn = math.cos(nu)
    sn = math.sin(nu)
    s_ap_nu = math.sin(argp + nu)
    c_ap_nu = math.cos(argp + nu)
    si = math.sin(i + 1e-10)   # inclination singularity left as-is
    ci = math.cos(i)
    sin_argp = math.sin(argp)
    cos_argp = math.cos(argp)
    h_now, k_now = hk_from_e_argp(e, argp)

    B = np.zeros((6, 3))

    # da/dt
    B[0, 0] = 2.0 * a ** 2 / h_ang * e * sn
    B[0, 1] = 2.0 * a ** 2 / h_ang * (p / r)

    # de/dt intermediates
    de_R = p * sn / h_ang
    de_S = ((p + r) * cn + r * e) / h_ang

    # dargp/dt coefficients, unnormalized (e cancels in the h/k form below)
    A_argp = -p * cn / h_ang
    B_argp = (p + r) * sn / h_ang
    C_argp = -r * s_ap_nu * ci / (h_ang * si)

    # dh/dt = de/dt * sin(argp) + k * dargp/dt
    B[1, 0] = de_R * sin_argp + cos_argp * A_argp
    B[1, 1] = de_S * sin_argp + cos_argp * B_argp
    B[1, 2] = k_now * C_argp

    # dk/dt = de/dt * cos(argp) - h * dargp/dt
    B[2, 0] = de_R * cos_argp - sin_argp * A_argp
    B[2, 1] = de_S * cos_argp - sin_argp * B_argp
    B[2, 2] = -h_now * C_argp

    # di/dt
    B[3, 2] = r * c_ap_nu / h_ang

    # dOmega/dt
    B[4, 2] = r * s_ap_nu / (h_ang * si)

    # dM/dt (still floored)
    e_floor = max(e, E_FLOOR_FOR_DM)
    factor = b_semi / (h_ang * e_floor * a)
    B[5, 0] = factor * ((p / r) * cn - 2.0 * e)
    B[5, 1] = -factor * ((p + r) / r) * sn

    return B


def _nu_to_M(nu: float, e: float) -> float:
    """True anomaly -> mean anomaly."""
    E = 2.0 * math.atan2(
        math.sqrt(max(0.0, 1.0 - e)) * math.sin(nu / 2.0),
        math.sqrt(max(0.0, 1.0 + e)) * math.cos(nu / 2.0),
    )
    M = E - e * math.sin(E)
    return M % (2.0 * math.pi)


def compute_B_matrix_orbit_averaged(el: MeanElements, n_samples: int = 360) -> np.ndarray:
    """Orbit-averaged Gauss-VOP B matrix for a constant RSW acceleration.

    Numerically averages the instantaneous B over one revolution, weighted
    by dt/dnu = r^2 / h. Use for continuous bias injection only; impulse
    tests should use the instantaneous form.
    """
    e, i, argp = el.ecc, el.inc, el.argp
    p = el.a * (1.0 - e * e)
    h_ang = math.sqrt(MU * p)

    B_sum = np.zeros((6, 3))
    w_sum = 0.0
    for k in range(n_samples):
        nu = 2.0 * math.pi * k / n_samples
        r = p / (1.0 + e * math.cos(nu))
        weight = r * r / h_ang
        el_nu = MeanElements(
            a=el.a, ecc=e, inc=i, raan=el.raan, argp=argp,
            M=_nu_to_M(nu, e),
        )
        B_sum += compute_B_matrix(el_nu) * weight
        w_sum += weight

    return B_sum / w_sum


def augmented_dynamics(
    x: np.ndarray,
    Cd: float,
    area: float,
    mass: float,
    epoch_jd: float,
    t_s: float,
    f107: float,
    kp: float,
    tau: float,
) -> np.ndarray:
    el = _elements_from_x(x)
    if el.a <= 0.0 or not np.isfinite(el.a):
        return np.zeros(9)

    w = x[6:9]
    e, argp = el.ecc, el.argp
    h_now, k_now = hk_from_e_argp(e, argp)

    alt = el.alt_m()
    rho = density(alt, f107, kp)
    br = brouwer_rates(el.a, el.ecc, el.inc)

    da_drag, de_drag = drag_rates(el.a, el.ecc, Cd, area, mass, rho)

    r_sun = sun_position_eci(epoch_jd, t_s)
    r_moon = moon_position_eci(epoch_jd, t_s)
    de_srp = srp_ecc_rate(el.a, area, mass, r_sun)
    di_3b, de_3b = third_body_rates(el.a, el.ecc, el.inc, r_moon, r_sun)

    d_a = da_drag
    de_dt_secular = de_drag + de_srp + de_3b
    d_inc = di_3b
    d_raan = br["d_raan"]
    dargp_dt_secular = br["d_argp"]
    dM_dt_secular = br["n"] + br["dn_j2"]

    d_h_secular = de_dt_secular * math.sin(argp) + k_now * dargp_dt_secular
    d_k_secular = de_dt_secular * math.cos(argp) - h_now * dargp_dt_secular

    B = compute_B_matrix(el)
    elem_rates_from_w = B @ w

    dxdt = np.zeros(9)
    dxdt[0] = d_a + elem_rates_from_w[0]
    dxdt[1] = d_h_secular + elem_rates_from_w[1]
    dxdt[2] = d_k_secular + elem_rates_from_w[2]
    dxdt[3] = d_inc + elem_rates_from_w[3]
    dxdt[4] = d_raan + elem_rates_from_w[4]
    dxdt[5] = dM_dt_secular + elem_rates_from_w[5]

    # FOGM decay (tau negative in backward filter)
    dxdt[6:9] = -w / tau

    return dxdt


def compute_stm(
    x: np.ndarray,
    dt: float,
    Cd: float,
    area: float,
    mass: float,
    epoch_jd: float,
    t_s: float,
    f107: float,
    kp: float,
    tau: float,
) -> np.ndarray:
    """State transition matrix Phi = I + A*dt via forward finite differences."""
    f0 = augmented_dynamics(x, Cd, area, mass, epoch_jd, t_s, f107, kp, tau)

    if not np.all(np.isfinite(f0)):
        return np.eye(9)

    A = np.zeros((9, 9))

    for j in range(9):
        h = max(abs(x[j]), config.STM_FD_FLOOR) * config.STM_FD_EPS
        x_plus = x.copy()
        x_plus[j] += h
        f_plus = augmented_dynamics(
            x_plus, Cd, area, mass, epoch_jd, t_s, f107, kp, tau
        )
        if not np.all(np.isfinite(f_plus)):
            continue
        diff = f_plus - f0
        if not np.all(np.isfinite(diff)):
            continue
        A[:, j] = diff / h

    Phi = np.eye(9) + A * dt
    return Phi


def compute_process_noise(
    dt: float,
    tau: float,
    q: float,
) -> np.ndarray:
    """Discrete process-noise covariance S for one step.

    q may be a scalar or a 3-vector [q_R, q_S, q_W].
    """
    S = np.zeros((9, 9))
    tau_abs = abs(tau)
    q_vec = np.atleast_1d(q)
    if q_vec.size == 1:
        q_vec = np.full(3, float(q_vec[0]))
    for i in range(3):
        if tau_abs > 1e-12:
            s_w = float(q_vec[i]) * tau_abs / 2.0 * (1.0 - math.exp(-2.0 * abs(dt) / tau_abs))
        else:
            s_w = float(q_vec[i]) * abs(dt)
        S[6 + i, 6 + i] = s_w
    return S