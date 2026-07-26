# Physical constants used throughout DRIFTS.
# Every value is tagged with its source. Don't change these without checking the reference.

import math

# --- Earth gravity [WGS84] ---
MU = 3.986004418e14         # m^3/s^2, Earth gravitational parameter
R_EARTH = 6.371e6           # mean radius, meters
RE_EQ = 6.378137e6          # equatorial radius, meters [WGS84]
OMEGA_EARTH = 7.2921150e-5  # Earth rotation rate, rad/s [WGS84]

# --- Moon / Sun gravity params [Montenbruck & Gill 2000] ---
MU_MOON = 4.902800066e12    # m^3/s^2
MU_SUN = 1.327124400e20     # m^3/s^2

# --- Gravity zonal harmonics [EGM96] ---
J2 = 1.08263e-3
J3 = -2.53265e-6
J4 = -1.61962e-6

# --- Time constants [IAU] ---
T_YEAR = 365.25 * 86400
T_SIDEREAL = 365.25636 * 86400     # sidereal year, seconds
T_CARRINGTON = 27.2753 * 86400     # synodic Carrington rotation, seconds

# --- Solar / atmospheric [IAU] ---
P_SRP = 4.56e-6              # solar radiation pressure at 1 AU, N/m^2
AU = 1.496e11                # meters
OBLIQUITY = 23.43929111 * math.pi / 180.0  # J2000 ecliptic obliquity, radians

# --- Reentry threshold ---
REENTRY_ALT = 100e3   # Karman line, meters

# --- TLE noise sigmas in MEAN ELEMENT space ---
# Derived from Vallado et al. 2006 Table 3 (RSW position/velocity residuals),
# propagated through Gauss VOP at a representative 600km/60deg reference orbit.
# CAVEAT: this reference-orbit linearization is an approximation - the true
# sigmas vary with altitude and inclination since Gauss VOP is orbit-dependent.
# Flagged here as a known limitation to revisit once Phase 1 sensitivity
# analysis tells us whether it matters.
SIGMA_A_M = 541.0     # meters, semi-major axis
SIGMA_ECC = 1.2e-5
SIGMA_INC = 1.5e-5    # radians
SIGMA_RAAN = 2.0e-4   # radians
SIGMA_ARGP = 3.0e-4   # radians
SIGMA_M = 8.5e-4      # radians

# --- Drag coefficient params [Moe & Moe 2005] ---
CD_MEAN = 2.2
CD_SIGMA = 0.2
CD_MIN = 1.8
CD_MAX = 2.8

# --- B* / Cd Ornstein-Uhlenbeck drift params [Vallado & Cefola 2012] ---
CD_TAU_S = 5.0 * 86400     # 5-day correlation timescale, seconds
CD_DRIFT_FRAC = 0.20       # 20% sigma per correlation time

# --- Detection floor ---
# Empirical, from Phase-1-style signal analysis: below this, SNR < 1 at
# 1 year for a 500-700km object. Treat as a starting prior, not a fixed law -
# Phase 1 sensitivity analysis is what re-derives this per altitude/duration.
DETECTION_FLOOR = 1e-11   # frags/m^3

# --- Debris population inclination distribution [Klinkrad 2006, Table 2.1] ---
_DEBRIS_INC_RAW = [
    (0,   0.02), (10,  0.02), (20,  0.03), (28,  0.05),
    (40,  0.05), (51,  0.08), (65,  0.07), (74,  0.05),
    (82,  0.06), (90,  0.05), (97,  0.15), (98,  0.15),
    (100, 0.05), (110, 0.04), (120, 0.03), (150, 0.03),
    (180, 0.07),
]
_tot = sum(f for _, f in _DEBRIS_INC_RAW)
DEBRIS_INC_POP = [(inc, frac / _tot) for inc, frac in _DEBRIS_INC_RAW]

# --- NRLMSISE-00-fit piecewise exponential atmosphere table ---
# (h0_km, rho0 [kg/m^3], scale_height_km), one row per 50km band, 200-1000km.
# Values follow the standard exponential atmosphere reference table used in
# astrodynamics texts (e.g. Vallado, "Fundamentals of Astrodynamics and
# Applications", low/mid solar activity band). This is the BASELINE only -
# the F10.7 and Kp corrections in atmosphere.py scale it further.
ATM_TABLE_KM = [
    (200, 2.789e-10, 37.105),
    (250, 7.248e-11, 45.546),
    (300, 2.418e-11, 53.628),
    (350, 9.518e-12, 53.298),
    (400, 3.725e-12, 58.515),
    (450, 1.585e-12, 60.828),
    (500, 6.967e-13, 63.822),
    (550, 3.152e-13, 71.835),
    (600, 1.454e-13, 88.667),
    (650, 6.947e-14, 105.643),
    (700, 3.386e-14, 125.259),
    (750, 1.905e-14, 132.780),
    (800, 1.058e-14, 140.987),
    (850, 6.523e-15, 158.865),
    (900, 4.145e-15, 168.792),
    (950, 2.755e-15, 178.234),
    (1000, 1.910e-15, 190.877),
]

# Sample TLEs used as fallback when no real TLE file is provided.
SAMPLE_TLES = [
    ("ISS (ZARYA)",
     "1 25544U 98067A   24001.50000000  .00001793  00000-0  40702-4 0  9990",
     "2 25544  51.6400 120.0000 0001234  45.0000 315.0000 15.49560000440000"),
    ("SENTINEL-2A",
     "1 40697U 15028A   24001.50000000  .00000100  00000-0  55000-5 0  9990",
     "2 40697  98.5700 100.0000 0001100  90.0000 270.0000 14.30818200450000"),
]