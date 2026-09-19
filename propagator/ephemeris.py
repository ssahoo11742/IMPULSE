# Low-precision Sun and Moon position vectors in ECI (equatorial) coordinates.
#
# PRECISION NOTE: these are the "low precision" formulas from Meeus, accurate
# to arcminutes for the Sun and roughly a degree for the Moon. That's plenty
# for our purposes - third-body and SRP terms are themselves small secular
# corrections, so a rough direction/distance is sufficient. Do NOT reuse this
# module anywhere precision ephemeris is actually needed.

import math
import numpy as np
from constants.constants import (
    AU, OBLIQUITY, SECONDS_PER_DAY,
    J2000_JD, SUN_MA_J2000_DEG, SUN_MA_RATE_DEG_PER_DAY,
    SUN_ML_J2000_DEG, SUN_ML_RATE_DEG_PER_DAY,
    SUN_EQ_CENTER_COEF1_DEG, SUN_EQ_CENTER_COEF2_DEG,
    SUN_DIST_MEAN_AU, SUN_DIST_ECC_TERM_AU, SUN_DIST_SMALL_TERM_AU,
    MOON_ML_J2000_DEG, MOON_ML_RATE_DEG_PER_DAY,
    MOON_MA_J2000_DEG, MOON_MA_RATE_DEG_PER_DAY,
    MOON_ARG_LAT_J2000_DEG, MOON_ARG_LAT_RATE_DEG_PER_DAY,
    MOON_LON_PERTURB_AMP_DEG, MOON_LAT_PERTURB_AMP_DEG,
    MOON_DIST_MEAN_M, MOON_DIST_VAR_AMP_M,
)


def _days_since_j2000(epoch_jd, elapsed_s):
    return (epoch_jd - J2000_JD) + elapsed_s / SECONDS_PER_DAY


def sun_position_eci(epoch_jd, elapsed_s):
    """Low-precision Sun position vector in ECI, meters. [Meeus ch. 25]"""
    d = _days_since_j2000(epoch_jd, elapsed_s)

    g = math.radians((SUN_MA_J2000_DEG + SUN_MA_RATE_DEG_PER_DAY * d) % 360.0)
    L = math.radians((SUN_ML_J2000_DEG + SUN_ML_RATE_DEG_PER_DAY * d) % 360.0)
    lam = (L + math.radians(SUN_EQ_CENTER_COEF1_DEG) * math.sin(g)
           + math.radians(SUN_EQ_CENTER_COEF2_DEG) * math.sin(2 * g))

    r_au = (SUN_DIST_MEAN_AU
            - SUN_DIST_ECC_TERM_AU * math.cos(g)
            - SUN_DIST_SMALL_TERM_AU * math.cos(2 * g))
    r = r_au * AU

    x = r * math.cos(lam)
    y = r * math.sin(lam) * math.cos(OBLIQUITY)
    z = r * math.sin(lam) * math.sin(OBLIQUITY)

    return np.array([x, y, z])


def moon_position_eci(epoch_jd, elapsed_s):
    """Low-precision Moon position vector in ECI, meters. Mean-orbit
    approximation only (no perturbation terms) - adequate for orbit-averaged
    third-body secular rates."""
    d = _days_since_j2000(epoch_jd, elapsed_s)

    L = math.radians((MOON_ML_J2000_DEG + MOON_ML_RATE_DEG_PER_DAY * d) % 360.0)
    M_moon = math.radians((MOON_MA_J2000_DEG + MOON_MA_RATE_DEG_PER_DAY * d) % 360.0)
    F = math.radians((MOON_ARG_LAT_J2000_DEG + MOON_ARG_LAT_RATE_DEG_PER_DAY * d) % 360.0)

    lam = L + math.radians(MOON_LON_PERTURB_AMP_DEG) * math.sin(M_moon)
    beta = math.radians(MOON_LAT_PERTURB_AMP_DEG) * math.sin(F)
    r = MOON_DIST_MEAN_M - MOON_DIST_VAR_AMP_M * math.cos(M_moon)

    x = r * math.cos(lam) * math.cos(beta)
    y = r * (math.sin(lam) * math.cos(beta) * math.cos(OBLIQUITY)
             - math.sin(beta) * math.sin(OBLIQUITY))
    z = r * (math.sin(lam) * math.cos(beta) * math.sin(OBLIQUITY)
             + math.sin(beta) * math.cos(OBLIQUITY))

    return np.array([x, y, z])