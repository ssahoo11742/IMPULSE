"""Validation ladder and synthetic measurement generation for the Phase-2 EKF.

Stages (reviewer-mandated order):
    2a  null test      — zero unmodeled acceleration
    2b  bias test      — constant injected RSW acceleration
    2c  impulse test   — single known debris strike
    2d  detection demo — full clean-vs-debris comparison
"""
from .ekf import _measurement_function 
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from .dynamics import hk_from_e_argp   
from propagator.orbital import (
    MeanElements, brouwer_rates, drag_rates, srp_ecc_rate,
    third_body_rates, mean_to_true_anomaly
)
from propagator.atmosphere import (
    density, sample_f107_phases, f107_at_time, sample_storm_events, kp_at_time
)
from propagator.ephemeris import sun_position_eci, moon_position_eci
from propagator.debris_impacts import apply_impact, gauss_vop
from constants.constants import (
    CD_MEAN, CD_SIGMA, CD_MIN, CD_MAX, CD_TAU_S, CD_DRIFT_FRAC,
    REENTRY_ALT, MU
)

from . import config
from .dynamics import compute_B_matrix, compute_B_matrix_orbit_averaged
from .ekf import _measurement_function, _classical_to_hk_measurement, EKF
from .smoother import fraser_potter_smoother
from .test_statistics import (
    compute_smoothed_accel_peak,
    compute_mahalanobis_distance,
    compute_mcreynolds,
)


# =============================================================================
# Helpers
# =============================================================================

def _step_truth(el, Cd, area, mass, epoch_jd, t_s, f107, kp, bias_w=None):
    """Single-step derivative for truth propagation (matches existing physics).

    Optionally adds a constant RSW acceleration ``bias_w`` via the Gauss-VOP
    B-matrix.
    """
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

    if bias_w is not None:
        # NOTE: orbit-averaged, not the instantaneous compute_B_matrix().
        # bias_w is a CONSTANT acceleration being Euler-stepped over dt_s
        # (typically a full day - many orbits at LEO altitudes), so the
        # rate used here needs to already be a secular/time-averaged rate,
        # the same way brouwer_rates/drag_rates/etc. are. Using the raw
        # instantaneous B matrix here was aliasing the periodic
        # within-orbit content into a large spurious daily signal that
        # swamped the actual injected bias (wrong sign, ~80x magnitude).
        B = compute_B_matrix_orbit_averaged(el)
        elem_rates = B @ np.asarray(bias_w)
        d_a += elem_rates[0]

        # B-matrix outputs [da/dt, dh/dt, dk/dt, di/dt, dOmega/dt, dM/dt].
        # _propagate_truth uses classical elements, so convert dh/dt, dk/dt
        # back to de/dt and dargp/dt via the chain rule.
        h_now, k_now = hk_from_e_argp(el.ecc, el.argp)
        e = el.ecc
        if e > 1e-12:
            de_bias = (h_now * elem_rates[1] + k_now * elem_rates[2]) / e
            dargp_bias = (k_now * elem_rates[1] - h_now * elem_rates[2]) / (e * e)
        else:
            de_bias = 0.0
            dargp_bias = 0.0

        d_ecc += de_bias
        d_inc += elem_rates[3]      # di/dt
        d_raan += elem_rates[4]     # dOmega/dt
        d_argp += dargp_bias
        d_M += elem_rates[5]

    return d_a, d_ecc, d_inc, d_raan, d_argp, d_M


def _propagate_truth(
    el0: MeanElements,
    epoch_jd: float,
    duration_s: float,
    dt_s: float,
    area: float,
    mass: float,
    f107_base: float = 150.0,
    Cd_base: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    bias_w: Optional[np.ndarray] = None,
    strike_time: Optional[float] = None,
    strike_dv_mag: float = 0.006,  # m/s, along -S; default matches the
                                    # previous hardcoded (1e-4 kg / mass=200) * 12e3 m/s
    vary_cd: bool = True,
    truth_substeps: int = 4,
) -> Dict:
    """Propagate a truth trajectory with optional constant bias or single strike.

    truth_substeps splits each dt_s Euler step into this many smaller
    sub-steps for the deterministic + bias rate integration (rates are
    recomputed each sub-step; environment (f107/kp) and Cd-drift/impact
    logic stay on the original dt_s cadence - those were designed around
    daily sampling). This mirrors EKF.predict's own hourly sub-stepping so
    the truth trajectory isn't systematically less accurate than what the
    filter is trying to fit against it. With compute_B_matrix_orbit_averaged
    already handling the bias term's secular-vs-instantaneous mismatch,
    this is now a secondary numerical-accuracy improvement rather than the
    primary fix - but it costs little and removes another source of
    truth/filter inconsistency. Set to 1 to recover the old single-step
    behavior.

    Returns a history dict with keys:
        t, a, e, i, Omega, omega, M, Cd
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

    hist = {
        "t": [], "a": [], "e": [], "i": [],
        "Omega": [], "omega": [], "M": [], "Cd": []
    }

    for _ in range(n_steps):
        if el.alt_m() < REENTRY_ALT:
            break

        f107 = f107_at_time(t, f107_phases, f_base=f107_base)
        kp = kp_at_time(t, storms)

        n_sub = max(1, truth_substeps)
        dt_sub = dt_s / n_sub
        for _ in range(n_sub):
            d_a, d_ecc, d_inc, d_raan, d_argp, d_M = _step_truth(
                el, Cd, area, mass, epoch_jd, t, f107, kp, bias_w
            )

            el.a += d_a * dt_sub
            el.ecc = max(0.0, el.ecc + d_ecc * dt_sub)
            el.inc += d_inc * dt_sub
            el.raan = (el.raan + d_raan * dt_sub) % (2 * math.pi)
            el.argp = (el.argp + d_argp * dt_sub) % (2 * math.pi)
            el.M += d_M * dt_sub

        # --- single strike injection ---
        if strike_time is not None and t <= strike_time < t + dt_s:
            v_circ = math.sqrt(MU / el.a)
            dv_mag = strike_dv_mag
            dv_rsw = np.array([0.0, -dv_mag, 0.0])
            nu, _ = mean_to_true_anomaly(el.M, el.ecc)
            da, de, di, dOm, darg = gauss_vop(el.a, el.ecc, el.inc, el.argp, nu, dv_rsw)
            el.a += da
            el.ecc = max(0.0, el.ecc + de)
            el.inc += di
            el.raan = (el.raan + dOm) % (2 * math.pi)
            el.argp = (el.argp + darg) % (2 * math.pi)

        if vary_cd:
            Cd += (1.0 / CD_TAU_S) * (Cd_base - Cd) * dt_s + cd_sigma_step * rng.normal()
            Cd = float(np.clip(Cd, CD_MIN, CD_MAX))
        
        t += dt_s

        hist["t"].append(t)
        hist["a"].append(el.a)
        hist["e"].append(el.ecc)
        hist["i"].append(el.inc)
        hist["Omega"].append(el.raan)
        hist["omega"].append(el.argp)
        hist["M"].append(el.M)
        hist["Cd"].append(Cd)

    return hist, storms, f107_phases


def generate_noisy_measurements(
    truth_hist: Dict,
    R_diag: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[float, np.ndarray]]:
    """Add Gaussian noise to truth mean elements.

    Returns list of (t, z) where z = [a, e, i, Omega, omega, M].
    M is kept unwrapped to match the filter's internal integration.
    """
    if rng is None:
        rng = np.random.default_rng()
    if R_diag is None:
        R_diag = config.R_DIAG_ELEMENTS

    n = len(truth_hist["t"])

    meas = []
    for i in range(n):
        z = np.array([
            truth_hist["a"][i],
            truth_hist["e"][i],
            truth_hist["i"][i],
            truth_hist["Omega"][i],
            truth_hist["omega"][i],
            truth_hist["M"][i],
        ])
        z += rng.normal(scale=np.sqrt(R_diag))
        meas.append((truth_hist["t"][i], z))

    return meas


def _build_initial_state(el0: MeanElements) -> np.ndarray:
    """9-state initial vector [elements, zero accel]."""
    h, k = hk_from_e_argp(el0.ecc, el0.argp)
    x0 = np.zeros(9)
    x0[0] = el0.a
    x0[1] = h
    x0[2] = k
    x0[3] = el0.inc
    x0[4] = el0.raan
    x0[5] = el0.M
    x0[6:9] = 0.0
    return x0


def _build_initial_covariance(
    R_diag: Optional[np.ndarray] = None,
    el0: Optional[MeanElements] = None,
) -> np.ndarray:
    """Diagonal initial covariance in (a, h, k, i, Omega, M) space."""
    if R_diag is None:
        R_diag = config.R_DIAG_ELEMENTS
    if el0 is not None:
        e = el0.ecc
        argp = el0.argp
        R_h = math.sin(argp)**2 * R_diag[1] + (e * math.cos(argp))**2 * R_diag[4]
        R_k = math.cos(argp)**2 * R_diag[1] + (e * math.sin(argp))**2 * R_diag[4]
    else:
        R_h = R_diag[1]
        R_k = R_diag[1]
    P0 = np.diag(np.concatenate([
        [R_diag[0], R_h, R_k, R_diag[2], R_diag[3], R_diag[5]],
        np.full(3, 1.0e-6)
    ]))
    return P0


def _run_filter_pair(
    measurements: List[Tuple[float, np.ndarray]],
    el0: MeanElements,
    Cd_base: float,
    area: float,
    mass: float,
    epoch_jd: float,
    tau: float,
    q: float,
    f107_base: float = 150.0,
    storms: Optional[List] = None,
    f107_phases: Optional[Any] = None,
) -> Dict:
    """Run forward EKF, backward EKF, and Fraser–Potter smoother.

    Returns dict with keys:
        times, fwd_states, fwd_covs, bwd_states, bwd_covs,
        sm_states, sm_covs, innovations
    """
    # --- forward pass ---
    x0 = _build_initial_state(el0)
    P0 = _build_initial_covariance(el0=el0)

    ekf_fwd = EKF(
        x0, P0, Cd_base, area, mass, epoch_jd,
        tau=tau, q=q, direction="forward"
    )

    duration_s = max(t for t, _ in measurements)
    if storms is None:
        rng_env = np.random.default_rng(42)
        storms = sample_storm_events(duration_s, rng_env)
    if f107_phases is None:
        rng_env = np.random.default_rng(42)
        f107_phases = sample_f107_phases(rng_env)

    fwd_states = []
    fwd_covs = []
    innovations = []
    prev_t = 0.0

    for t, z in measurements:
        dt = t - prev_t
        f107 = f107_at_time(prev_t, f107_phases, f_base=f107_base)
        kp = kp_at_time(prev_t, storms)
        ekf_fwd.predict(dt, f107, kp)
        ekf_fwd.update(z, config.R_DIAG_ELEMENTS)
        fwd_states.append(ekf_fwd.x.copy())
        fwd_covs.append(ekf_fwd.P.copy())
        
        innovations.append(_classical_to_hk_measurement(z) - _measurement_function(ekf_fwd.x))
        prev_t = t

    # --- backward pass ---
       # --- backward pass ---
    x0_bwd = fwd_states[-1].copy()
    P0_bwd = fwd_covs[-1].copy() * 100.0

    ekf_bwd = EKF(
        x0_bwd, P0_bwd, Cd_base, area, mass, epoch_jd,
        tau=tau, q=q, direction="backward"
    )

    bwd_apriori_states = []
    bwd_apriori_covs = []
    prev_t = measurements[-1][0]

    for t, z in reversed(measurements):
        dt = t - prev_t
        # First reversed step is the final epoch: no prediction needed
        if abs(dt) > 1e-12:
            # Use prev_t (the later time) for env, matching forward-pass convention
            f107 = f107_at_time(prev_t, f107_phases, f_base=f107_base)
            kp = kp_at_time(prev_t, storms)
            ekf_bwd.predict(dt, f107, kp)
        
        bwd_apriori_states.append(ekf_bwd.x.copy())
        bwd_apriori_covs.append(ekf_bwd.P.copy())
        
        ekf_bwd.update(z, config.R_DIAG_ELEMENTS)
        prev_t = t
        
    # --- smoother ---
    sm_states, sm_covs = fraser_potter_smoother(
        fwd_states, fwd_covs,
        list(reversed(bwd_apriori_states)),
        list(reversed(bwd_apriori_covs)),
    )

    times = [t for t, _ in measurements]

    return {
        "times": times,
        "fwd_states": fwd_states,
        "fwd_covs": fwd_covs,
        "bwd_states": list(reversed(bwd_apriori_states)),
        "bwd_covs": list(reversed(bwd_apriori_covs)),
        "sm_states": sm_states,
        "sm_covs": sm_covs,
        "innovations": innovations,
    }


# =============================================================================
# Validation ladder
# =============================================================================

def _default_el0(alt_km: float = 800.0, inc_deg: float = 40.0) -> MeanElements:
    """Build a default initial mean element set."""
    RE = 6371e3
    a0 = (alt_km * 1000 + RE) / (1 - 0.001)
    return MeanElements(
        a=a0, ecc=0.001, inc=math.radians(inc_deg),
        raan=0.0, argp=0.0, M=0.0
    )


def run_null_test(
    duration_days: int = 30,
    dt_s: float = 86400.0,
    tau: Optional[float] = None,
    q: Optional[float] = None,
    seed: int = 42,
) -> Dict:
    """2a: Zero unmodeled acceleration. Smoothed accel should stay near zero."""
    rng = np.random.default_rng(seed)
    el0 = _default_el0()
    epoch_jd = 2460000.5
    duration_s = duration_days * 86400.0
    area, mass = 1.0, 200.0
    Cd_base = float(np.clip(rng.normal(CD_MEAN, CD_SIGMA), CD_MIN, CD_MAX))

    truth, storms, f107_phases = _propagate_truth(
        el0, epoch_jd, duration_s, dt_s, area, mass,
        Cd_base=Cd_base, rng=rng
    )
    meas = generate_noisy_measurements(truth, rng=rng)

    tau = tau if tau is not None else config.DEFAULT_TAU_S
    q = q if q is not None else config.DEFAULT_Q

    result = _run_filter_pair(
        meas, el0, Cd_base, area, mass, epoch_jd, tau, q,
        storms=storms, f107_phases=f107_phases
    )

    margin_s = 7.0 * 86400.0
    times_arr = np.array(result["times"])
    mask = (times_arr >= margin_s) & (times_arr <= times_arr[-1] - margin_s)
    sm_trimmed = [s for s, m in zip(result["sm_states"], mask) if m]
    times_trimmed = times_arr[mask].tolist()

    stats = compute_smoothed_accel_peak(sm_trimmed, times_trimmed)

    return {
        "test": "null",
        "peak_accel_m_s2": stats["peak_norm"],
        "mean_offpeak_m_s2": stats["mean_offpeak"],
        "snr": stats["snr"],
        "passed": stats["peak_norm"] < 1.0e-4,
        "result": result,
        "stats": stats,
    }


def run_bias_test(
    duration_days: int = 30,
    dt_s: float = 86400.0,
    bias_w: np.ndarray = None,
    tau: Optional[float] = None,
    q: Optional[float] = None,
    seed: int = 43,
) -> Dict:
    """2b: Constant injected RSW acceleration. Filter should recover it."""
    if bias_w is None:
        bias_w = np.array([0.0, 1.0e-6, 0.0])

    rng = np.random.default_rng(seed)
    el0 = _default_el0()
    epoch_jd = 2460000.5
    duration_s = duration_days * 86400.0
    area, mass = 1.0, 200.0
    Cd_base = float(np.clip(rng.normal(CD_MEAN, CD_SIGMA), CD_MIN, CD_MAX))

    truth, storms, f107_phases = _propagate_truth(
        el0, epoch_jd, duration_s, dt_s, area, mass,
        Cd_base=Cd_base, rng=rng, bias_w=bias_w, vary_cd=False
    )
    meas = generate_noisy_measurements(truth, rng=rng)

    tau = tau if tau is not None else config.DEFAULT_TAU_S
    q_scalar = q if q is not None else config.DEFAULT_Q

    # Only w_S is secularly observable with 1-day cadence.
    # Suppress random-walk of the unobservable R and W channels.
    q_vec = np.array([0.0, q_scalar, 0.0])

    result = _run_filter_pair(
        meas, el0, Cd_base, area, mass, epoch_jd, tau, q_vec,
        storms=storms, f107_phases=f107_phases
    )

    # Steady-state estimated w (interior only, excluding edge transients).
    #
    # This margin was originally a flat "30 days of steps" - wrong on two
    # counts. (1) For runs <=~60 days that's an empty slice -> crash (see
    # prior fix). (2) More fundamentally: w_R/w_W here have q=0, so the
    # ONLY way they return to zero after the early-epoch burn-in transient
    # (the filter overshooting on the unobservable-with-1-sample/day R,W
    # channels before it has enough data to disambiguate them) is the
    # deterministic FOGM decay exp(-t/tau). If tau is comparable to or
    # larger than the trim window, you're averaging the transient itself,
    # not a converged value - this was silently inflating estimated_w for
    # R/W by orders of magnitude at tau=1e6s (a ~10-day window against an
    # 11.6-day tau is < 1 e-folding). So the margin needs to cover several
    # tau, not just a fixed calendar window.
    n_pts = len(result["fwd_states"])
    margin_burn_in = int(math.ceil(3.0 * tau / dt_s))    # >=3 e-foldings
    margin_floor = max(3, int(5.0 * 86400.0 / dt_s))     # basic settle time, independent of tau
    desired_margin = max(margin_burn_in, margin_floor)
    margin = max(1, min(desired_margin, n_pts // 3))
    if desired_margin > n_pts // 3:
        raise ValueError(
            f"run_bias_test: duration_days={duration_days} (dt_s={dt_s}) gives "
            f"only {n_pts} measurement(s), but tau={tau:.3e}s needs a burn-in "
            f"margin of ~{margin_burn_in} steps (>=3*tau) to reach steady "
            f"state, leaving no interior window to average over. Use a "
            f"longer duration_days (duration >= ~10*tau is a reasonable "
            f"floor) or a shorter tau."
        )
    w_est = np.array([s[6:9] for s in result["fwd_states"][margin:-margin]])
    w_mean = np.mean(w_est, axis=0)
    w_err = abs(w_mean[1] - bias_w[1]) / abs(bias_w[1])

    return {
        "test": "bias",
        "injected_w": bias_w,
        "estimated_w": w_mean,
        "relative_error": w_err,
        "passed": w_err < 0.10,
        "result": result,
    }


def run_impulse_test(
    duration_days: int = 30,
    dt_s: float = 86400.0,
    strike_time_days: float = 15.0,
    strike_dv_mag: float = 0.006,
    tau: Optional[float] = None,
    q: Optional[float] = None,
    seed: int = 44,
) -> Dict:
    """2c: Single known debris strike. Smoothed accel should peak at strike epoch."""
    rng = np.random.default_rng(seed)
    el0 = _default_el0()
    epoch_jd = 2460000.5
    duration_s = duration_days * 86400.0
    area, mass = 1.0, 200.0
    Cd_base = float(np.clip(rng.normal(CD_MEAN, CD_SIGMA), CD_MIN, CD_MAX))
    strike_time = strike_time_days * 86400.0

    truth, storms, f107_phases = _propagate_truth(
        el0, epoch_jd, duration_s, dt_s, area, mass,
        Cd_base=Cd_base, rng=rng, strike_time=strike_time,
        strike_dv_mag=strike_dv_mag,
    )
    meas = generate_noisy_measurements(truth, rng=rng)

    tau = tau if tau is not None else config.DEFAULT_TAU_S
    q = q if q is not None else config.DEFAULT_Q

    result = _run_filter_pair(
        meas, el0, Cd_base, area, mass, epoch_jd, tau, q,
        storms=storms, f107_phases=f107_phases
    )
    stats = compute_smoothed_accel_peak(
        result["sm_states"], result["times"],
        strike_window=(strike_time - dt_s, strike_time + dt_s)
    )

    offpeak_ok = stats["mean_offpeak"] < 0.1 * stats["peak_norm"]
    timing_ok = abs(stats["peak_time"] - strike_time) <= dt_s

    return {
        "test": "impulse",
        "strike_time_s": strike_time,
        "peak_time_s": stats["peak_time"],
        "peak_accel_m_s2": stats["peak_norm"],
        "mean_offpeak_m_s2": stats["mean_offpeak"],
        "snr": stats["snr"],
        "passed": timing_ok and offpeak_ok,
        "result": result,
        "stats": stats,
    }


def run_single_detection(
    duration_days: int = 30,
    dt_s: float = 86400.0,
    rho_debris: float = 1.0e-10,
    tau: Optional[float] = None,
    q: Optional[float] = None,
    seed: int = 45,
) -> Dict:
    """Full demo: clean vs debris, noisy measurements, filter, smoother, stats."""
    from propagator.propagator import propagate_clean, propagate_debris

    rng = np.random.default_rng(seed)
    el0 = _default_el0()
    epoch_jd = 2460000.5
    duration_s = duration_days * 86400.0
    area, mass = 1.0, 200.0
    Cd_base = float(np.clip(rng.normal(CD_MEAN, CD_SIGMA), CD_MIN, CD_MAX))
    f107_base = 150.0

    clean_res = propagate_clean(
        el0, epoch_jd, duration_s, dt_s, area, mass,
        f107_base=f107_base, Cd_base=Cd_base, rng=rng, record_history=True
    )

    rng_debris = np.random.default_rng(seed)
    debris_res, total_impacts = propagate_debris(
        el0, epoch_jd, duration_s, dt_s, area, mass, rho_debris,
        f107_base=f107_base, Cd_base=Cd_base, rng=rng_debris, record_history=True
    )

    meas = generate_noisy_measurements(debris_res.history, rng=rng)

    tau = tau if tau is not None else config.DEFAULT_TAU_S
    q = q if q is not None else config.DEFAULT_Q

    result = _run_filter_pair(meas, el0, Cd_base, area, mass, epoch_jd, tau, q)

    sm = result["sm_states"]
    times = result["times"]
    fwd = result["fwd_states"]
    bwd = result["bwd_states"]
    smc = result["sm_covs"]
    fc = result["fwd_covs"]
    bc = result["bwd_covs"]

    accel_stats = compute_smoothed_accel_peak(sm, times)
    d_mh = compute_mahalanobis_distance(fwd, fc, bwd, bc, smc)
    mcr = compute_mcreynolds(fwd, fc, sm, smc)

    return {
        "test": "detection",
        "total_impacts": total_impacts,
        "accel_stats": accel_stats,
        "mahalanobis_peak": float(np.max(d_mh)),
        "mcreynolds_peak": float(np.max(mcr["scalar"])),
        "result": result,
        "clean_history": clean_res.history,
        "debris_history": debris_res.history,
    }