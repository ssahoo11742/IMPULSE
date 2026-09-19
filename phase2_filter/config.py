"""Phase 2 filter configuration — all tunables in one place.

Units match the existing codebase (metres, radians, seconds).
"""

import numpy as np

# -----------------------------------------------------------------------------
# FOGM unmodeled-acceleration tuning
# -----------------------------------------------------------------------------
# Time constant (seconds).  Bennett ~1 s for GPS cadence.
# For daily TLE cadence (dt = 86400 s) start near 1e5 s (~1 day).
DEFAULT_TAU_S: float = 1.0e6

# Process-noise spectral density per RSW channel.
# Units: [acceleration]^2 * [time] = (m/s^2)^2 * s = m^2 / s^3.
#
# SIZING RULE: the FOGM state's own steady-state prior std is
# sqrt(q * tau / 2). If that's >> the smallest acceleration you actually
# want to detect, the filter has no reason to prefer "constant bias" over
# "random walk" and will fit measurement noise instead of the real signal
# (confirmed via run_bias_test grid search - the old 1e-10 gave a steady
# -state std of ~2.2e-3 m/s^2 against a 1e-6 m/s^2 injected bias, a
# >2000x mismatch, and the filter converged to noise: relative error in
# the thousands of percent, with the wrong sign). Re-derive whenever tau
# or your target detection floor changes:
#     q ~= 2 * target_accel_std^2 / tau
# The value below was grid-searched against a 1e-6 m/s^2 bias test at
# tau=1e5 s / daily cadence (see run_bias_test); 5e-17 to 1e-16 passed at
# 30 days. Longer baselines (e.g. 365 days) still need q even smaller /
# tau retuned - this has NOT been re-derived for arbitrary durations yet.
DEFAULT_Q: float = 5.0e-17

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