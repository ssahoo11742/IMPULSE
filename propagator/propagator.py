# Clean-run propagator: integrates Brouwer secular + drag + SRP + third-body
# rates forward in time. No debris forcing here yet - that's propagator's
# debris-run counterpart, added once this baseline is validated.
#
# INTEGRATION NOTE: we use simple Euler stepping at dt=1 day, not RK4. The
# rates here are already orbit-averaged/secular (slow compared to the orbital
# period itself), and the environmental drivers (F10.7, sun/moon position)
# change slowly compared to 1 day, so higher-order integration buys little.
# This is a deliberate simplification, flagged for revisiting if Phase 1
# sensitivity analysis shows integration error matters at the ρ-detection
# level we care about.

import math
import numpy as np
from dataclasses import dataclass, field
from .orbital import MeanElements, brouwer_rates, drag_rates, srp_ecc_rate, third_body_rates
from .atmosphere import density, sample_f107_phases, f107_at_time, sample_storm_events, kp_at_time
from .ephemeris import sun_position_eci, moon_position_eci
from .debris_impacts import compute_vmf_kappa, sample_impact_count, apply_impact
from constants.constants import CD_MEAN, CD_SIGMA, CD_MIN, CD_MAX, CD_TAU_S, CD_DRIFT_FRAC, REENTRY_ALT, MU


@dataclass
class PropagationResult:
    final_elements: MeanElements
    final_Cd: float
    history: dict = field(default_factory=dict)   # arrays vs time, for diagnostics
    reentered: bool = False


def _step_rates(el, Cd, area, mass, epoch_jd, t_s, f107, kp):
    alt = el.alt_m()
    rho = density(alt, f107, kp)

    br = brouwer_rates(el.a, el.ecc, el.inc)
    da_drag, de_drag = drag_rates(el.a, el.ecc, Cd, area, mass, rho)

    r_sun = sun_position_eci(epoch_jd, t_s)
    r_moon = moon_position_eci(epoch_jd, t_s)

    de_srp = srp_ecc_rate(el.a, area, mass, r_sun)
    di_3b, de_3b = third_body_rates(el.a, el.ecc, el.inc, r_moon, r_sun)

    d_a = da_drag
    d_ecc = de_drag + de_srp + de_3b
    d_inc = di_3b
    d_raan = br["d_raan"]
    d_argp = br["d_argp"]
    d_M = br["n"] + br["dn_j2"]

    return d_a, d_ecc, d_inc, d_raan, d_argp, d_M


def propagate_clean(el0, epoch_jd, duration_s, dt_s, area, mass,
                     f107_base=150.0, Cd_base=None, rng=None, record_history=True):
    """
    Propagate a satellite forward with NO debris forcing - gravity harmonics,
    drag, SRP, and lunisolar third-body effects only. Returns final elements
    plus (optionally) a time history for diagnostics/validation.
    """
    if rng is None:
        rng = np.random.default_rng()
    if Cd_base is None:
        Cd_base = float(np.clip(rng.normal(CD_MEAN, CD_SIGMA), CD_MIN, CD_MAX))

    el = el0.copy()
    Cd = Cd_base
    t = 0.0
    n_steps = int(duration_s / dt_s)

    storms = sample_storm_events(duration_s, rng)
    f107_phases = sample_f107_phases(rng)
    cd_sigma_step = CD_DRIFT_FRAC * Cd_base * math.sqrt(2 * dt_s / CD_TAU_S)

    hist = {"t": [], "a": [], "ecc": [], "inc": [], "Cd": []} if record_history else None
    reentered = False

    for _ in range(n_steps):
        if el.alt_m() < REENTRY_ALT:
            reentered = True
            break

        f107 = f107_at_time(t, f107_phases, f_base=f107_base)
        kp = kp_at_time(t, storms)

        d_a, d_ecc, d_inc, d_raan, d_argp, d_M = _step_rates(
            el, Cd, area, mass, epoch_jd, t, f107, kp)

        el.a += d_a * dt_s
        el.ecc = max(0.0, el.ecc + d_ecc * dt_s)
        el.inc += d_inc * dt_s
        el.raan = (el.raan + d_raan * dt_s) % (2 * math.pi)
        el.argp = (el.argp + d_argp * dt_s) % (2 * math.pi)
        el.M = (el.M + d_M * dt_s) % (2 * math.pi)

        # Cd Ornstein-Uhlenbeck drift (Euler-Maruyama)
        Cd += (1.0 / CD_TAU_S) * (Cd_base - Cd) * dt_s + cd_sigma_step * rng.normal()
        Cd = float(np.clip(Cd, CD_MIN, CD_MAX))

        t += dt_s

        if record_history:
            hist["t"].append(t)
            hist["a"].append(el.a)
            hist["ecc"].append(el.ecc)
            hist["inc"].append(el.inc)
            hist["Cd"].append(Cd)

    return PropagationResult(final_elements=el, final_Cd=Cd, history=hist or {}, reentered=reentered)


def propagate_debris(el0, epoch_jd, duration_s, dt_s, area, mass, rho_debris,
                      f107_base=150.0, Cd_base=None, rng=None, record_history=True):
    """
    Propagate a satellite forward with the SAME deterministic physics as
    propagate_clean (gravity harmonics, drag, SRP, lunisolar), PLUS stochastic
    debris impacts at density rho_debris (fragments/m^3, referenced to the
    1mm minimum fragment size - see debris_impacts.LC_MIN_M).

    Call this and propagate_clean from the SAME el0/epoch_jd/rng-seed pair to
    get the differential (clean vs debris) signal the whole project is built
    around - the two runs share every deterministic force, so the only
    difference between their final states is the accumulated debris kicks.
    """
    if rng is None:
        rng = np.random.default_rng()
    if Cd_base is None:
        Cd_base = float(np.clip(rng.normal(CD_MEAN, CD_SIGMA), CD_MIN, CD_MAX))

    el = el0.copy()
    Cd = Cd_base
    t = 0.0
    n_steps = int(duration_s / dt_s)

    storms = sample_storm_events(duration_s, rng)
    f107_phases = sample_f107_phases(rng)
    cd_sigma_step = CD_DRIFT_FRAC * Cd_base * math.sqrt(2 * dt_s / CD_TAU_S)

    hist = {"t": [], "a": [], "ecc": [], "inc": [], "Cd": [], "n_impacts": []} if record_history else None
    reentered = False
    total_impacts = 0

    for _ in range(n_steps):
        if el.alt_m() < REENTRY_ALT:
            reentered = True
            break

        f107 = f107_at_time(t, f107_phases, f_base=f107_base)
        kp = kp_at_time(t, storms)

        d_a, d_ecc, d_inc, d_raan, d_argp, d_M = _step_rates(
            el, Cd, area, mass, epoch_jd, t, f107, kp)

        el.a += d_a * dt_s
        el.ecc = max(0.0, el.ecc + d_ecc * dt_s)
        el.inc += d_inc * dt_s
        el.raan = (el.raan + d_raan * dt_s) % (2 * math.pi)
        el.argp = (el.argp + d_argp * dt_s) % (2 * math.pi)
        el.M = (el.M + d_M * dt_s) % (2 * math.pi)

        # --- debris impacts this step ---
        v_circ = math.sqrt(MU / el.a)
        kappa, _mean_cos, flux_sum_unit = compute_vmf_kappa(el.inc)
        v_rel_mean = flux_sum_unit * v_circ   # scale the unit-v_circ flux sum by the real speed
        n_impacts = sample_impact_count(rho_debris, area, v_rel_mean, dt_s, rng)
        for _ in range(n_impacts):
            apply_impact(el, mass, el.inc, v_circ, kappa, rng)
        total_impacts += n_impacts

        # Cd Ornstein-Uhlenbeck drift (Euler-Maruyama)
        Cd += (1.0 / CD_TAU_S) * (Cd_base - Cd) * dt_s + cd_sigma_step * rng.normal()
        Cd = float(np.clip(Cd, CD_MIN, CD_MAX))

        t += dt_s

        if record_history:
            hist["t"].append(t)
            hist["a"].append(el.a)
            hist["ecc"].append(el.ecc)
            hist["inc"].append(el.inc)
            hist["Cd"].append(Cd)
            hist["n_impacts"].append(n_impacts)

    return PropagationResult(final_elements=el, final_Cd=Cd, history=hist or {}, reentered=reentered), total_impacts