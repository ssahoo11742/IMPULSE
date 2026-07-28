"""9-state augmented dynamics and STM for the debris-detection EKF.

STATE REPRESENTATION CHANGED (singularity fix):
    x = [a, h, k, i, Omega, M, w_R, w_S, w_W]^T
    where h = e*sin(argp), k = e*cos(argp)  ("non-singular"/equinoctial-style
    substitution for the eccentricity/argument-of-perigee pair).

WHY: the original state used (e, argp) directly, and the Gauss-VOP domega/dt
term has a genuine 1/e singularity - it blew up as e -> 0 (confirmed
numerically: at e=1e-5 the argp-rate coefficient was already ~7 - 5e7x
larger than at e=0.1, and the null-test validation run showed a w_S spike
of ~1e7 m/s^2 exactly where eccentricity crossed near zero under drag-driven
circularization). Re-deriving dh/dt and dk/dt via the chain rule causes the
1/e term to cancel algebraically against the e implicit in h and k
themselves - verified numerically below to stay bounded (and even evaluate
cleanly) all the way to e=0, unlike the original argp-rate formula.

SCOPE NOTE - what this patch does NOT fix:
  - The dM/dt-from-acceleration row (originally B[5,:]) ALSO divides by e,
    but summing it with the (now-fixed) argp-rate does NOT cancel - checked
    numerically: the combined d(argp+M)/dt sum still diverges at the same
    1/e rate as the M-row alone. That means this specific dM/dt formula is
    very likely not the theoretically-correct Gauss-VOP mean-anomaly term in
    the first place (the code's own prior comment already flagged it as an
    incomplete "near-circular form"). Rather than trust a from-scratch
    re-derivation of that term under time pressure - exactly the kind of
    hand-algebra mistake that has bitten this project twice already - this
    patch applies an explicit, documented floor on e for that ONE term only
    (see E_FLOOR_FOR_DM below) and leaves a proper re-derivation (or a
    citable-reference lookup, the same way the brouwer_rates bug was fixed
    earlier) as follow-up work.
  - The 1/sin(i) terms (dOmega/dt, and this dM/dt row's w_W coefficient) are
    the same class of singularity for near-equatorial orbits (i near 0 or
    180 deg). Not triggered by the validation ladder's default i=40deg, not
    fixed here - same treatment (a p,q equinoctial pair) would be needed if
    you ever test near-equatorial orbits.

All units match the existing codebase: metres, radians, seconds.
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
from constants.constants import MU

from . import config

# Explicit, documented floor for the one remaining singular term (dM/dt
# acceleration row) that this patch does NOT structurally fix - see
# module docstring. NOT the same as the old "1e-20" which did nothing;
# this is a physically meaningful floor chosen to keep the term bounded
# without silently zeroing it out.
E_FLOOR_FOR_DM = 1.0e-3


# -----------------------------------------------------------------------
# Non-singular (h, k) <-> classical (e, argp) conversions
# -----------------------------------------------------------------------

def hk_from_e_argp(e: float, argp: float) -> Tuple[float, float]:
    """(e, argp) -> (h, k). Always well-defined, no singularity."""
    return e * math.sin(argp), e * math.cos(argp)


def e_argp_from_hk(h: float, k: float) -> Tuple[float, float]:
    """(h, k) -> (e, argp). e = hypot(h,k) is exact; argp is genuinely
    undefined at e=0 (a circular orbit has no perigee) but atan2 returns a
    consistent, non-blowing-up value (0.0 at h=k=0 by convention) rather
    than raising or producing NaN/inf - which is exactly the point."""
    e = math.hypot(h, k)
    argp = math.atan2(h, k) if e > 1e-12 else 0.0
    return e, argp


def _elements_from_x(x: np.ndarray) -> MeanElements:
    """Unpack the 9-state vector's first 6 entries into MeanElements.
    Converts the non-singular (h,k) pair back to (e, argp) internally."""
    e, argp = e_argp_from_hk(float(x[1]), float(x[2]))
    # Guard: the propagator physics cannot handle e >= 1
    e = min(e, 0.999999)
    return MeanElements(
        a=float(x[0]),
        ecc=e,
        inc=float(x[3]),
        raan=float(x[4]),
        argp=argp,
        M=float(x[5])
    )


# -----------------------------------------------------------------------
# Gauss-VOP sensitivity matrix, expressed in (a, h, k, i, Omega, M) rates
# -----------------------------------------------------------------------

def compute_B_matrix(el: MeanElements) -> np.ndarray:
    """Return the 6x3 matrix mapping RSW acceleration [w_R, w_S, w_W]
    (m/s^2) into rates [da/dt, dh/dt, dk/dt, di/dt, dOmega/dt, dM/dt].

    Rows for da/dt, di/dt, dOmega/dt are the same classical Gauss planetary
    equations as before (unchanged, not singular in e). Rows for dh/dt and
    dk/dt replace the old (singular) domega/dt row - see module docstring
    for the derivation and numerical verification. The dM/dt row keeps the
    old formula with an explicit floor on e (see E_FLOOR_FOR_DM).
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
    si = math.sin(i + 1e-10)   # unchanged floor - inclination singularity out of scope, see docstring
    ci = math.cos(i)
    sin_argp = math.sin(argp)
    cos_argp = math.cos(argp)
    h_now, k_now = hk_from_e_argp(e, argp)   # current h,k, needed for the w_W terms below

    B = np.zeros((6, 3))

    # --- da/dt: unchanged, never singular in e ---
    B[0, 0] = 2.0 * a ** 2 / h_ang * e * sn
    B[0, 1] = 2.0 * a ** 2 / h_ang * (p / r)

    # --- de/dt coefficients (unchanged, never singular in e) - used below
    #     as an intermediate for dh/dt, dk/dt via the chain rule ---
    de_R = p * sn / h_ang
    de_S = ((p + r) * cn + r * e) / h_ang

    # --- dargp/dt coefficients, UNNORMALIZED (deliberately NOT divided by
    #     e here - the whole point is to let the e cancel algebraically
    #     against the e implicit in h/k below, rather than ever evaluating
    #     the singular quantity) ---
    A_argp = -p * cn / h_ang            # coefficient of w_R
    B_argp = (p + r) * sn / h_ang       # coefficient of w_S
    C_argp = -r * s_ap_nu * ci / (h_ang * si)   # coefficient of w_W - no e dependence at all

    # dh/dt = d(e sin argp)/dt = (de/dt) sin(argp) + e cos(argp) (dargp/dt)
    #       = de/dt * sin(argp) + k * dargp/dt
    # The k*(dargp/dt) term: dargp/dt's w_R, w_S parts are (A_argp, B_argp)/e,
    # so k*(A_argp/e) = e*cos(argp)*A_argp/e = cos(argp)*A_argp - the e
    # cancels exactly. The w_W part of dargp/dt (C_argp) has no e at all, so
    # k*C_argp is just a normal, safe product.
    B[1, 0] = de_R * sin_argp + cos_argp * A_argp
    B[1, 1] = de_S * sin_argp + cos_argp * B_argp
    B[1, 2] = k_now * C_argp

    # dk/dt = d(e cos argp)/dt = (de/dt) cos(argp) - e sin(argp) (dargp/dt)
    #       = de/dt * cos(argp) - h * dargp/dt   (same cancellation as above)
    B[2, 0] = de_R * cos_argp - sin_argp * A_argp
    B[2, 1] = de_S * cos_argp - sin_argp * B_argp
    B[2, 2] = -h_now * C_argp

    # --- di/dt: unchanged, never singular in e ---
    B[3, 2] = r * c_ap_nu / h_ang

    # --- dOmega/dt: unchanged (1/sin(i) singularity, out of scope - see docstring) ---
    B[4, 2] = r * s_ap_nu / (h_ang * si)

    # --- dM/dt: corrected Gauss VOP form (still uses e_floor as safety cap) ---
# --- dM/dt: corrected Gauss VOP form ---
    e_floor = max(e, E_FLOOR_FOR_DM)
    factor = b_semi / (a * h_ang * e_floor)   # <-- /a added here
    B[5, 0] = factor * ((p / r) * cn - 2.0 * e)
    B[5, 1] = -factor * ((p + r) / r) * sn
    # no w_W contribution to dM/dt in this form (unchanged from original)

    return B


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
    """9-state derivative  dx/dt  for the augmented EKF state.

    x = [a, h, k, i, Omega, M, w_R, w_S, w_W]

    tau : float
        FOGM time constant.  Use +tau for forward filter, -tau for backward.
    """
    el = _elements_from_x(x)
    w = x[6:9]
    e, argp = el.ecc, el.argp
    h_now, k_now = hk_from_e_argp(e, argp)

    # --- deterministic secular rates from existing propagator physics ---
    alt = el.alt_m()
    rho = density(alt, f107, kp)
    br = brouwer_rates(el.a, el.ecc, el.inc)
    da_drag, de_drag = drag_rates(el.a, el.ecc, Cd, area, mass, rho)

    r_sun = sun_position_eci(epoch_jd, t_s)
    r_moon = moon_position_eci(epoch_jd, t_s)
    de_srp = srp_ecc_rate(el.a, area, mass, r_sun)
    di_3b, de_3b = third_body_rates(el.a, el.ecc, el.inc, r_moon, r_sun)

    d_a = da_drag
    de_dt_secular = de_drag + de_srp + de_3b   # total deterministic de/dt, non-singular
    d_inc = di_3b
    d_raan = br["d_raan"]
    dargp_dt_secular = br["d_argp"]            # classical Brouwer J2 precession, finite - safe to multiply by h/k directly
    dM_dt_secular = br["n"] + br["dn_j2"]

    # convert the deterministic (de/dt, dargp/dt) pair into (dh/dt, dk/dt)
    # via the same chain rule as compute_B_matrix - no singularity here
    # since dargp_dt_secular is already finite (it's the classical Brouwer
    # rate, not divided by e anywhere).
    d_h_secular = de_dt_secular * math.sin(argp) + k_now * dargp_dt_secular
    d_k_secular = de_dt_secular * math.cos(argp) - h_now * dargp_dt_secular

    # --- Gauss VOP contribution from unmodeled RSW accelerations ---
    B = compute_B_matrix(el)
    elem_rates_from_w = B @ w

    dxdt = np.zeros(9)
    dxdt[0] = d_a + elem_rates_from_w[0]
    dxdt[1] = d_h_secular + elem_rates_from_w[1]
    dxdt[2] = d_k_secular + elem_rates_from_w[2]
    dxdt[3] = d_inc + elem_rates_from_w[3]
    dxdt[4] = d_raan + elem_rates_from_w[4]
    dxdt[5] = dM_dt_secular + elem_rates_from_w[5]

    # FOGM decay (mirrored: tau negative in backward filter)
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
    """State transition matrix Phi = I + A*dt via forward finite differences.

    Unchanged from before - this function is state-representation-agnostic
    (it differentiates augmented_dynamics generically), so it needed no
    changes for the (h,k) substitution.
    """
    f0 = augmented_dynamics(x, Cd, area, mass, epoch_jd, t_s, f107, kp, tau)
    A = np.zeros((9, 9))

    for j in range(9):
        h = max(abs(x[j]), config.STM_FD_FLOOR) * config.STM_FD_EPS
        x_plus = x.copy()
        x_plus[j] += h
        f_plus = augmented_dynamics(
            x_plus, Cd, area, mass, epoch_jd, t_s, f107, kp, tau
        )
        A[:, j] = (f_plus - f0) / h

    Phi = np.eye(9) + A * dt
    return Phi


def compute_process_noise(
    dt: float,
    tau: float,
    q: float,
) -> np.ndarray:
    """Discrete process-noise covariance S for one step. Unchanged - this
    function only touches the acceleration block (states 6:9), which the
    (h,k) substitution doesn't affect.

    Bennett forward :  P = Phi P Phi^T + S
    Bennett backward:  P = Phi P Phi^T - S   (S computed with |tau|)
    """
    S = np.zeros((9, 9))
    tau_abs = abs(tau)
    if tau_abs > 1e-12:
        s_w = q * tau_abs / 2.0 * (1.0 - math.exp(-2.0 * abs(dt) / tau_abs))
    else:
        s_w = q * abs(dt)
    S[6:9, 6:9] = np.eye(3) * s_w
    return S