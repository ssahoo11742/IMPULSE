# Atmospheric density model: piecewise-exponential baseline (fit to NRLMSISE-00)
# with solar (F10.7) and geomagnetic (Kp) storm corrections layered on top.

import math
import numpy as np
from constants.constants import (
    ATM_TABLE_KM, T_YEAR, T_CARRINGTON, SECONDS_PER_DAY,
    F107_BASELINE, F107_ANNUAL_AMP, F107_CARRINGTON_AMP,
    STORM_RATE_PER_YEAR, STORM_PEAK_KP_MEAN, STORM_PEAK_KP_SIGMA,
    STORM_DURATION_DAYS_MEAN, STORM_DURATION_DAYS_SIGMA,
    STORM_DURATION_DAYS_MIN, QUIET_KP, STORM_PROFILE_PEAK_FRAC,
    F107_DENSITY_SCALE_DENOM, DENSITY_SOLAR_MAX_MULT, KP_DENSITY_EXP_COEF,
)


def base_density(alt_m):
    alt_km = alt_m / 1000.0
    table = ATM_TABLE_KM

    h_min = table[0][0]
    h_max = table[-1][0]
    alt_km = max(h_min, min(alt_km, h_max))

    for i in range(len(table) - 1):
        h0, rho0, H = table[i]
        h1 = table[i + 1][0]
        if h0 <= alt_km < h1:
            return rho0 * math.exp(-(alt_km - h0) / H)

    h0, rho0, H = table[-1]
    return rho0 * math.exp(-(alt_km - h0) / H)


def sample_f107_phases(rng):
    """Draw fixed random phases for the annual + Carrington F10.7 cycles.
    Call ONCE per simulation run, then reuse the same phases at every
    timestep via f107_at_time - re-randomizing per-step would turn a smooth
    periodic signal into uncorrelated noise."""
    return rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi)


def f107_at_time(t_s, phases, f_base=F107_BASELINE):
    """F10.7 solar flux index at time t_s (seconds from simulation start),
    given fixed phases from sample_f107_phases. Two sinusoids: an annual
    cycle and the ~27-day Carrington rotation, amplitudes from NOAA solar
    cycle 24 statistics."""
    phi1, phi2 = phases
    return (f_base
            + F107_ANNUAL_AMP * math.sin(2 * math.pi * t_s / T_YEAR + phi1)
            + F107_CARRINGTON_AMP * math.sin(2 * math.pi * t_s / T_CARRINGTON + phi2))


def sample_storm_events(duration_s, rng, rate_per_year=STORM_RATE_PER_YEAR):
    """
    Sample geomagnetic storm events as a Poisson process over the simulation
    duration. Returns list of (onset_time_s, peak_kp, duration_s).
    Calibrated to NOAA SWPC solar-cycle-24 statistics.
    """
    duration_years = duration_s / T_YEAR
    n_storms = rng.poisson(rate_per_year * duration_years)

    events = []
    for _ in range(n_storms):
        onset = rng.uniform(0, duration_s)
        peak_kp = np.clip(rng.normal(STORM_PEAK_KP_MEAN, STORM_PEAK_KP_SIGMA), 0, 9)
        dur_days = max(STORM_DURATION_DAYS_MIN,
                       rng.normal(STORM_DURATION_DAYS_MEAN, STORM_DURATION_DAYS_SIGMA))
        events.append((onset, peak_kp, dur_days * SECONDS_PER_DAY))

    return sorted(events, key=lambda e: e[0])


def kp_at_time(t_s, storm_events):
    """Kp index at time t_s, given a list of storm events."""
    kp = QUIET_KP
    for onset, peak_kp, dur_s in storm_events:
        if onset <= t_s <= onset + dur_s:
            # simple triangular profile: rise to peak at 1/3 duration, decay after
            frac = (t_s - onset) / dur_s
            shape = 1.0 - abs(frac - STORM_PROFILE_PEAK_FRAC) / max(frac, 1 - frac, STORM_PROFILE_PEAK_FRAC)
            kp = max(kp, QUIET_KP + (peak_kp - QUIET_KP) * max(0.0, shape))
    return kp


def density(alt_m, f107, kp):
    """
    Full density model: base * solar correction * storm correction.
    rho_solar(F10.7) ~ log-linear, gives ~3x variation solar-min to solar-max.
    rho_storm(Kp) = exp(0.32 * Kp)  [Picone et al. 2002]
    """
    rho = base_density(alt_m)
    rho *= math.exp((f107 - F107_BASELINE) / F107_DENSITY_SCALE_DENOM * math.log(DENSITY_SOLAR_MAX_MULT))
    rho *= math.exp(KP_DENSITY_EXP_COEF * (kp - QUIET_KP))
    return rho