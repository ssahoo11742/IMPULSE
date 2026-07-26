# Debris impact model.
#
# Three pieces, matching the project's original design:
#   1. WHEN does an impact happen - Poisson process, rate = rho * A_sat * v_rel * dt
#   2. WHAT hits - fragment size/mass from the NASA Standard Breakup Model (NSBM)
#   3. WHICH DIRECTION - anisotropic von Mises-Fisher distribution centered on
#      the anti-velocity direction, since most catalogued debris shares
#      roughly the satellite's orbital plane family (prograde LEO) and the
#      dominant collision geometry is head-on.
#
# KEY DESIGN CHOICE: the impact direction is sampled directly in the RSW frame
# (Radial/along-track/cross-track), not ECI. RSW's along-track (S) axis IS the
# satellite's velocity direction, so "anti-velocity" is just (0,-1,0) in RSW
# coordinates - no need to ever leave mean-element space to apply this.

import math
import numpy as np
from constants.constants import DEBRIS_INC_POP
from .orbital import mean_to_true_anomaly, gauss_vop

# NSBM fragment size distribution [Johnson et al. 2001]
# Minimum fragment size this debris density ρ is defined relative to.
# NOT specified in the original design doc - this is a modeling choice: we
# anchor ρ at 1mm, since the whole point of DRIFTS is inferring the density of
# fragments BELOW the ~10cm radar tracking floor. Revisit if Phase 1
# sensitivity analysis suggests results are sensitive to this choice.
LC_MIN_M = 1.0e-3


def sample_fragment_lc(rng, n=1):
    """Sample characteristic length(s) Lc (meters) from the NSBM power law
    via inverse CDF: N(Lc>L) ~ L^-1.71, so F(L) = 1 - (L/Lmin)^-1.71."""
    u = rng.uniform(0.0, 1.0, size=n)
    return LC_MIN_M * (1.0 - u) ** (-1.0 / 1.71)


def fragment_area_to_mass(Lc):
    """NSBM area-to-mass ratio, piecewise power law [Johnson et al. 2001, Table 2].
    Returns A/M in m^2/kg. IMPORTANT: Lc goes into the formula in METERS - the
    1.67mm/96mm regime boundary is conventionally described in mm, but the
    log10(Lc) term itself is meters. Using mm here previously produced
    unphysical A/M values (>1000 m^2/kg for a 10cm fragment; real debris
    tops out around 10-40 m^2/kg even for thin foil)."""
    if Lc < 1.67e-3:
        log_am = -0.3 * math.log10(Lc) - 1.4
    else:
        log_am = 0.97 * math.log10(Lc) + 1.149
    return 10.0 ** log_am


def fragment_mass(Lc):
    """Fragment mass (kg) from characteristic length via area-to-mass ratio."""
    area = math.pi * (Lc / 2.0) ** 2
    am_ratio = fragment_area_to_mass(Lc)
    return area / am_ratio


def _rel_velocity_and_angle(sat_inc, debris_inc_rad, v_circ):
    """For two circular co-altitude orbits at inclinations sat_inc and
    debris_inc_rad: relative velocity magnitude and angle of that relative
    velocity vector from the satellite's anti-velocity direction."""
    di = sat_inc - debris_inc_rad
    v_rel = 2.0 * v_circ * abs(math.sin(di / 2.0))
    # theta is scale-invariant in v_circ (cancels in atan2), kept explicit for clarity
    theta = math.atan2(math.sin(di), (1.0 - math.cos(di)))
    return v_rel, theta


def compute_vmf_kappa(sat_inc):
    """
    Derive the vMF concentration parameter kappa for the impact-direction
    distribution, from the catalogued debris population's inclination mix
    [Klinkrad 2006, Table 2.1 -> constants.DEBRIS_INC_POP].

    Flux-weighted mean cosine of impact angle, then invert the Langevin
    function L(kappa) = coth(kappa) - 1/kappa = <cos theta> numerically.
    Also returns the flux-weighted mean relative velocity, reused by the
    impact-rate calculation so the "how often" and "which direction" pieces
    are internally consistent with the same population data.
    """
    num_cos = 0.0
    flux_sum = 0.0
    for inc_deg, frac in DEBRIS_INC_POP:
        v_rel, theta = _rel_velocity_and_angle(sat_inc, math.radians(inc_deg), v_circ=1.0)
        weight = frac * v_rel   # flux ~ population fraction * relative speed
        num_cos += weight * math.cos(theta)
        flux_sum += weight

    mean_cos = num_cos / flux_sum if flux_sum > 0 else 0.0
    kappa = _invert_langevin(mean_cos)
    return kappa, mean_cos, flux_sum   # flux_sum here is dimensionless (v_circ=1); scale by real v_circ later


def _invert_langevin(target_cos, lo=1e-6, hi=200.0):
    def langevin(k):
        return 1.0 / math.tanh(k) - 1.0 / k

    if target_cos <= langevin(lo):
        return lo
    if target_cos >= langevin(hi):
        return hi
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if langevin(mid) < target_cos:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def sample_vmf_direction(kappa, rng):
    """
    Sample a unit vector from a von Mises-Fisher distribution on S^2 centered
    on (0, -1, 0) in RSW coordinates (i.e. anti-along-track), concentration
    kappa. Wood's (1994) rejection algorithm, specialized to p=3 (so the
    Beta((p-1)/2,(p-1)/2) draw reduces to a plain Uniform(0,1)).
    """
    if kappa < 1e-6:
        w = rng.uniform(-1.0, 1.0)
    else:
        b = -kappa + math.sqrt(kappa ** 2 + 1.0)   # (p-1)=2 case
        x0 = (1.0 - b) / (1.0 + b)
        c = kappa * x0 + 2.0 * math.log(1.0 - x0 ** 2)
        while True:
            z = rng.uniform(0.0, 1.0)
            w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
            u = rng.uniform(0.0, 1.0)
            if kappa * w + 2.0 * math.log(1.0 - x0 * w) - c >= math.log(u):
                break

    phi = rng.uniform(0.0, 2 * math.pi)
    r = math.sqrt(max(0.0, 1.0 - w ** 2))
    # local frame: z-axis (last component) aligned with mu
    local = np.array([r * math.cos(phi), r * math.sin(phi), w])

    # rotate so local z-axis maps onto mu = (0,-1,0) in RSW (R,S,W) ordering
    mu = np.array([0.0, -1.0, 0.0])
    return _rotate_z_to(local, mu)


def _rotate_z_to(vec, target):
    """Rotate `vec` (defined in a frame whose z-axis is [0,0,1]) into a frame
    whose z-axis is `target`. Simple axis-angle rotation; target is a unit vector."""
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, target)
    s = np.linalg.norm(v)
    c = np.dot(z, target)
    if s < 1e-12:
        return vec * np.sign(c) if c != 0 else vec   # already aligned (or anti-aligned)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
    return R @ vec


def sample_impact_velocity(sat_inc, v_circ, rng):
    """Sample a single impact's relative velocity magnitude, drawing the
    'other' debris object's inclination from the flux-weighted population
    distribution (higher relative speed -> more flux -> more likely to be
    the one that hits)."""
    incs = [d[0] for d in DEBRIS_INC_POP]
    fracs = np.array([d[1] for d in DEBRIS_INC_POP])
    v_rels = np.array([_rel_velocity_and_angle(sat_inc, math.radians(i), v_circ)[0] for i in incs])
    weights = fracs * v_rels
    weights = weights / weights.sum()
    idx = rng.choice(len(incs), p=weights)
    return v_rels[idx]


def sample_impact_count(rho, area, v_rel_mean, dt_s, rng):
    """Poisson-distributed number of impacts this timestep."""
    lam = rho * area * v_rel_mean * dt_s
    return rng.poisson(lam)


def apply_impact(el, mass_sat, sat_inc, v_circ, kappa, rng):
    """
    Sample one impact event and apply its velocity kick to the orbital
    elements in place (mutates el). Returns the fragment mass and relative
    velocity used, for diagnostics.
    """
    Lc = sample_fragment_lc(rng)[0]
    m_frag = fragment_mass(Lc)
    v_rel = sample_impact_velocity(sat_inc, v_circ, rng)

    direction_rsw = sample_vmf_direction(kappa, rng)   # unit vector, (R,S,W)
    dv_mag = (m_frag / mass_sat) * v_rel
    dv_rsw = direction_rsw * dv_mag

    nu, _E = mean_to_true_anomaly(el.M, el.ecc)
    da, de, di, dOm, dargp = gauss_vop(el.a, el.ecc, el.inc, el.argp, nu, dv_rsw)

    el.a += da
    el.ecc = max(0.0, el.ecc + de)
    el.inc += di
    el.raan = (el.raan + dOm) % (2 * math.pi)
    el.argp = (el.argp + dargp) % (2 * math.pi)

    return m_frag, v_rel