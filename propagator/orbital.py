# Orbital mechanics: Brouwer secular rates, Gauss variation-of-parameters, mean elements.
#
# Design choice carried over from the previous build: work in MEAN ELEMENT
# SPACE the whole time, not Cartesian ECI + RK4. Real TLEs are themselves a fit
# in mean-element space (that's what SGP4 consumes/produces), so propagating
# secular rates directly keeps us in the same representation as the real data
# and is far cheaper than integrating Cartesian state every 60s for a year.
# Trade-off: secular/orbit-averaged rates don't capture short-period
# (within-one-orbit) oscillations - only intended for the fixed points we
# actually observe TLEs at, not the fine structure in between.

import math
import numpy as np
from dataclasses import dataclass
from constants.constants import MU, RE_EQ, J2, J4, R_EARTH, MU_MOON, MU_SUN, P_SRP, AU


@dataclass
class MeanElements:
    a: float      # semi-major axis, meters
    ecc: float    # eccentricity
    inc: float    # inclination, radians
    raan: float   # right ascension of ascending node, radians
    argp: float   # argument of perigee, radians
    M: float      # mean anomaly, radians

    def mean_motion(self):
        return math.sqrt(MU / max(self.a, 1e6) ** 3)   # rad/s

    def alt_m(self):
        return self.a - R_EARTH   # mean altitude, circular approximation

    def copy(self):
        return MeanElements(self.a, self.ecc, self.inc, self.raan, self.argp, self.M)


def brouwer_rates(a, ecc, inc):
    """
    Secular drift rates from J2 (+ J4) zonal harmonics, rad/s.

    J2 terms verified against two independent sources and cross-checked
    numerically against real TLE data (see build notes):
      RAAN:  d(Omega)/dt = -(3/2) n J2 (Re/p)^2 cos(i)
      argp:  d(omega)/dt =  (3/4) n J2 (Re/p)^2 (4 - 5 sin^2 i)
      mean motion correction: dn = (3/2) n J2 (Re/p)^2 sqrt(1-e^2) (1 - 1.5 sin^2 i)
    [Vallado, Fundamentals of Astrodynamics and Applications; consistent with
    Kozai 1959 first-order secular terms]

    J4 terms are carried over from the earlier implementation and have NOT
    been independently re-verified the way the J2 terms above were - J4 is
    ~1000x smaller than J2 (see project README), so this is a low-priority
    fix, but treat these specific coefficients as provisional.
    """
    n = math.sqrt(MU / a ** 3)
    p = a * (1.0 - ecc ** 2)
    eta = math.sqrt(1.0 - ecc ** 2)

    ci = math.cos(inc)
    ci2 = ci * ci
    si2 = 1.0 - ci2

    re_p2 = (RE_EQ / p) ** 2

    # --- verified J2 terms ---
    d_raan = -1.5 * n * J2 * re_p2 * ci
    d_argp = 0.75 * n * J2 * re_p2 * (4.0 - 5.0 * si2)
    dn_j2 = 1.5 * n * J2 * re_p2 * eta * (1.0 - 1.5 * si2)

    # --- unverified J4 correction, carried over (small relative to J2) ---
    g4 = -3.0 / 8.0 * J4 * (RE_EQ / p) ** 4
    d_raan += n * g4 * (5.0 / 4.0) * ci * (3.0 * si2 - 4.0)
    d_argp += n * g4 * (35.0 / 8.0) * (1.0 - 11.0 / 5.0 * ci2 - 8.0 / 7.0 * ci2 ** 2) / eta

    return {"n": n, "d_raan": d_raan, "d_argp": d_argp, "dn_j2": dn_j2}


def drag_rates(a, ecc, Cd, area, mass, rho_atm):
    """Secular da/dt, de/dt from atmospheric drag [King-Hele 1987, eq 2.16/2.17]."""
    v_circ = math.sqrt(MU / a)
    beta = Cd * area / mass   # m^2/kg

    da_dt = -2.0 * a * beta * rho_atm * v_circ   # negative: orbit shrinks
    de_dt = -beta * rho_atm * v_circ * ecc / 2.0

    return da_dt, de_dt


def srp_ecc_rate(a, area, mass, r_sun_vec):
    """Orbit-averaged SRP eccentricity-pumping rate [Montenbruck & Gill, eq 3.80]."""
    rs = float(np.linalg.norm(r_sun_vec))
    if rs < 1e6:
        return 0.0

    n = math.sqrt(MU / a ** 3)
    cr = 1.3   # mean reflectivity [Moe & Moe 2005]
    a_srp = P_SRP * cr * area / mass * (AU / rs) ** 2

    sun_xy_frac = float(np.linalg.norm(r_sun_vec[:2])) / rs   # simplified orbit-plane projection
    return (3.0 / 2.0) * a_srp / (n * a) * sun_xy_frac


def third_body_rates(a, ecc, inc, r_moon, r_sun):
    """Kozai-type orbit-averaged lunisolar rates on inclination/eccentricity
    [Montenbruck & Gill, eq 3.91-3.93]."""
    n = math.sqrt(MU / a ** 3)

    total_di = 0.0
    total_de = 0.0

    for mu_body, r_body in [(MU_MOON, r_moon), (MU_SUN, r_sun)]:
        rb = float(np.linalg.norm(r_body))
        if rb < 1e6:
            continue

        n_body = math.sqrt(mu_body / rb ** 3)
        alpha = (a / rb) ** 3

        di = (15.0 / 8.0) * (n_body ** 2 / n) * alpha * ecc * math.sin(2.0 * inc)
        de = (15.0 / 16.0) * (n_body ** 2 / n) * alpha * math.sqrt(1.0 - ecc ** 2) * math.sin(2.0 * inc)

        total_di += di
        total_de += de

    return total_di, total_de


def mean_to_true_anomaly(M, ecc, tol=1e-10, max_iter=50):
    """Solve Kepler's equation M = E - e*sin(E) for eccentric anomaly E via
    Newton's method, then convert to true anomaly nu. Needed at the moment of
    a debris impact, since Gauss VOP requires true anomaly, not mean anomaly."""
    M = M % (2 * math.pi)
    E = M if ecc < 0.8 else math.pi   # better starting guess for high-e orbits
    for _ in range(max_iter):
        dE = (E - ecc * math.sin(E) - M) / (1.0 - ecc * math.cos(E))
        E -= dE
        if abs(dE) < tol:
            break
    nu = 2.0 * math.atan2(math.sqrt(1 + ecc) * math.sin(E / 2.0),
                          math.sqrt(1 - ecc) * math.cos(E / 2.0))
    return nu, E


def gauss_vop(a, ecc, inc, argp, nu, dv_rsw):
    """Gauss variation-of-parameters: converts an instantaneous RSW velocity
    impulse (radial, along-track, cross-track) into osculating-element
    changes [Montenbruck & Gill, eq 2.38-2.42]. This is how a debris impact
    kick gets translated into a change in (a, ecc, inc, raan, argp)."""
    p = a * (1.0 - ecc ** 2)
    r = p / (1.0 + ecc * math.cos(nu))
    h = math.sqrt(MU * p)

    cn = math.cos(nu)
    sn = math.sin(nu)

    dv_R = float(dv_rsw[0])
    dv_S = float(dv_rsw[1])
    dv_W = float(dv_rsw[2])

    da = 2.0 * a ** 2 / h * (ecc * sn * dv_R + (p / r) * dv_S)
    de = (p * sn * dv_R + ((p + r) * cn + r * ecc) * dv_S) / h
    di = r * math.cos(argp + nu) / h * dv_W
    dOm = r * math.sin(argp + nu) / (h * math.sin(inc + 1e-10)) * dv_W
    darg = ((-p * cn * dv_R + (p + r) * sn * dv_S) / (h * (ecc + 1e-20))
            - r * math.sin(argp + nu) * math.cos(inc) / (h * math.sin(inc + 1e-10)) * dv_W)

    return da, de, di, dOm, darg