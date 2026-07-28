"""Phase 2 filter configuration — all tunables in one place.

Units match the existing codebase (metres, radians, seconds).
"""

import numpy as np

# -----------------------------------------------------------------------------
# FOGM unmodeled-acceleration tuning
# -----------------------------------------------------------------------------
# Time constant (seconds).  Bennett ~1 s for GPS cadence.
# For daily TLE cadence (dt = 86400 s) start near 1e5 s (~1 day).
DEFAULT_TAU_S: float = 1.0e5

# Process-noise spectral density per RSW channel.
# Units: [acceleration]^2 * [time] = (m/s^2)^2 * s = m^2 / s^3.
# Start conservative; grid-search later.
DEFAULT_Q: float = 1.0e-10

# -----------------------------------------------------------------------------
# Measurement noise diagonal for synthetic / TLE mean elements
# Order: [a, e, i, Omega, omega, M]
# -----------------------------------------------------------------------------
R_DIAG_ELEMENTS = np.array([
    100.0,      # a  [m]      — TLE semi-major axis noise
    1.0e-4,     # e  [1]
    1.0e-4,     # i  [rad]
    1.0e-4,     # Omega [rad]
    1.0e-4,     # omega [rad]
    1.0e-3,     # M  [rad]   — noisiest due to SGP4 short-periodic content
])

# -----------------------------------------------------------------------------
# STM finite-difference settings
# -----------------------------------------------------------------------------
STM_FD_EPS: float = 1.0e-7

# Floor for finite-difference perturbation (prevents h=0 on zero states).
STM_FD_FLOOR: float = 1.0e-6 