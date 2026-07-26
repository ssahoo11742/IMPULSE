# Atmospheric density model: piecewise-exponential baseline (fit to NRLMSISE-00)
# with solar (F10.7) and geomagnetic (Kp) storm corrections layered on top.

import math
import numpy as np
from constants.constants import ATM_TABLE_KM, T_YEAR, T_CARRINGTON


def base_density(alt_m):
    """Piecewise exponential density from the reference table, kg/m^3."""
    alt_km = alt_m / 1000.0
    table = ATM_TABLE_KM

    if alt_km <= table[0][0]:
        h0, rho0, H = table[0]
        return rho0 * math.exp(-(alt_km - h0) / H)
    if alt_km >= table[-1][0]:
        h0, rho0, H = table[-1]
        return rho0 * math.exp(-(alt_km - h0) / H)

    # find the band this altitude falls in, use ITS h0/rho0/H (matches how
    # NRLMSISE-00 fits are actually tabulated - each band anchored at its own base)
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


def f107_at_time(t_s, phases, f_base=150.0):
    """F10.7 solar flux index at time t_s (seconds from simulation start),
    given fixed phases from sample_f107_phases. Two sinusoids: an annual
    cycle and the ~27-day Carrington rotation, amplitudes from NOAA solar
    cycle 24 statistics."""
    phi1, phi2 = phases
    return (f_base
            + 20.0 * math.sin(2 * math.pi * t_s / T_YEAR + phi1)
            + 15.0 * math.sin(2 * math.pi * t_s / T_CARRINGTON + phi2))


def sample_storm_events(duration_s, rng, rate_per_year=40.0):
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
        peak_kp = np.clip(rng.normal(5.5, 1.0), 0, 9)
        dur_days = max(0.1, rng.normal(1.5, 0.5))
        events.append((onset, peak_kp, dur_days * 86400))

    return sorted(events, key=lambda e: e[0])


def kp_at_time(t_s, storm_events):
    """Kp index at time t_s, given a list of storm events. Quiet background Kp ~ 2."""
    kp = 2.0
    for onset, peak_kp, dur_s in storm_events:
        if onset <= t_s <= onset + dur_s:
            # simple triangular profile: rise to peak at 1/3 duration, decay after
            frac = (t_s - onset) / dur_s
            shape = 1.0 - abs(frac - 0.33) / max(frac, 1 - frac, 0.33)
            kp = max(kp, 2.0 + (peak_kp - 2.0) * max(0.0, shape))
    return kp


def density(alt_m, f107, kp):
    """
    Full density model: base * solar correction * storm correction.
    ρ_solar(F10.7) ~ log-linear, gives ~3x variation solar-min to solar-max.
    ρ_storm(Kp) = exp(0.32 * Kp)  [Picone et al. 2002]
    """
    rho = base_density(alt_m)
    rho *= math.exp((f107 - 150.0) / 100.0 * math.log(3.0))
    rho *= math.exp(0.32 * (kp - 2.0))   # normalized so quiet Kp=2 gives no extra boost
    return rho