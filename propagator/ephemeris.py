# Low-precision Sun and Moon position vectors in ECI (equatorial) coordinates.
#
# PRECISION NOTE: these are the "low precision" formulas from Meeus, accurate
# to arcminutes for the Sun and roughly a degree for the Moon. That's plenty
# for our purposes - third-body and SRP terms are themselves small secular
# corrections, so a rough direction/distance is sufficient. Do NOT reuse this
# module anywhere precision ephemeris is actually needed.

import math
import numpy as np
from constants.constants import AU, OBLIQUITY


def _days_since_j2000(epoch_jd, elapsed_s):
    return (epoch_jd - 2451545.0) + elapsed_s / 86400.0


def sun_position_eci(epoch_jd, elapsed_s):
    """Low-precision Sun position vector in ECI, meters. [Meeus ch. 25]"""
    d = _days_since_j2000(epoch_jd, elapsed_s)

    g = math.radians((357.529 + 0.98560028 * d) % 360.0)          # mean anomaly
    L = math.radians((280.459 + 0.98564736 * d) % 360.0)          # mean longitude
    lam = L + math.radians(1.915) * math.sin(g) + math.radians(0.020) * math.sin(2 * g)

    r_au = 1.00014 - 0.01671 * math.cos(g) - 0.00014 * math.cos(2 * g)
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

    L = math.radians((218.316 + 13.176396 * d) % 360.0)     # mean longitude
    M_moon = math.radians((134.963 + 13.064993 * d) % 360.0)  # mean anomaly
    F = math.radians((93.272 + 13.229350 * d) % 360.0)       # argument of latitude

    lam = L + math.radians(6.289) * math.sin(M_moon)
    beta = math.radians(5.128) * math.sin(F)
    r = 385000.6e3 - 20905.4e3 * math.cos(M_moon)   # meters

    x = r * math.cos(lam) * math.cos(beta)
    y = r * (math.sin(lam) * math.cos(beta) * math.cos(OBLIQUITY) - math.sin(beta) * math.sin(OBLIQUITY))
    z = r * (math.sin(lam) * math.cos(beta) * math.sin(OBLIQUITY) + math.sin(beta) * math.cos(OBLIQUITY))

    return np.array([x, y, z])