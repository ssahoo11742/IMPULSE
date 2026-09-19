# Physical constants used throughout DRIFTS.
# Every value is tagged with its source. Don't change these without checking the reference.

import math

# --- Time [IAU] ---
SECONDS_PER_DAY = 86400.0
T_YEAR = 365.25 * SECONDS_PER_DAY
T_SIDEREAL = 365.25636 * SECONDS_PER_DAY     # sidereal year, seconds
T_CARRINGTON = 27.2753 * SECONDS_PER_DAY     # synodic Carrington rotation, seconds

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
CD_TAU_S = 5.0 * SECONDS_PER_DAY     # 5-day correlation timescale, seconds
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



# --- Ephemeris: low-precision Sun (Meeus ch. 25) ---
J2000_JD = 2451545.0
SUN_MA_J2000_DEG = 357.529
SUN_MA_RATE_DEG_PER_DAY = 0.98560028
SUN_ML_J2000_DEG = 280.459
SUN_ML_RATE_DEG_PER_DAY = 0.98564736
SUN_EQ_CENTER_COEF1_DEG = 1.915
SUN_EQ_CENTER_COEF2_DEG = 0.020
SUN_DIST_MEAN_AU = 1.00014
SUN_DIST_ECC_TERM_AU = 0.01671
SUN_DIST_SMALL_TERM_AU = 0.00014

# --- Ephemeris: low-precision Moon mean orbit ---
MOON_ML_J2000_DEG = 218.316
MOON_ML_RATE_DEG_PER_DAY = 13.176396
MOON_MA_J2000_DEG = 134.963
MOON_MA_RATE_DEG_PER_DAY = 13.064993
MOON_ARG_LAT_J2000_DEG = 93.272
MOON_ARG_LAT_RATE_DEG_PER_DAY = 13.229350
MOON_LON_PERTURB_AMP_DEG = 6.289
MOON_LAT_PERTURB_AMP_DEG = 5.128
MOON_DIST_MEAN_M = 385000.6e3
MOON_DIST_VAR_AMP_M = 20905.4e3

# --- Atmospheric model parameters ---
F107_BASELINE = 150.0
F107_ANNUAL_AMP = 20.0
F107_CARRINGTON_AMP = 15.0
STORM_RATE_PER_YEAR = 40.0
STORM_PEAK_KP_MEAN = 5.5
STORM_PEAK_KP_SIGMA = 1.0
STORM_DURATION_DAYS_MEAN = 1.5
STORM_DURATION_DAYS_SIGMA = 0.5
STORM_DURATION_DAYS_MIN = 0.1
QUIET_KP = 2.0
STORM_PROFILE_PEAK_FRAC = 1.0 / 3.0
F107_DENSITY_SCALE_DENOM = 100.0
DENSITY_SOLAR_MAX_MULT = 3.0
KP_DENSITY_EXP_COEF = 0.32

# --- Orbital mechanics numerical parameters ---
MEAN_MOTION_A_MIN_M = 1e6
KEPLER_TOL = 1e-10
KEPLER_MAX_ITER = 50
KEPLER_E_GUESS_THRESHOLD = 0.8
GAUSS_VOP_INC_EPS = 1e-10
GAUSS_VOP_ECC_EPS = 1e-20
SRP_CR_MEAN = 1.3
THIRDBODY_MIN_DIST_M = 1e6

# --- Debris impact model (NSBM & vMF) ---
LC_MIN_M = 1.0e-3
NSBM_POWER_LAW_EXP = 1.71
NSBM_AM_REGIME_BOUNDARY_M = 1.67e-3
NSBM_AM_SMALL_A = -0.3
NSBM_AM_SMALL_B = -1.4
NSBM_AM_LARGE_A = 0.97
NSBM_AM_LARGE_B = 1.149
VMF_KAPPA_MIN = 1e-6
VMF_KAPPA_MAX = 200.0
VMF_KAPPA_BISECT_ITER = 80
ROTATION_SINGULARITY_EPS = 1e-12

\subsection{Orbital Null Model}

\subsubsection{Force Model}

The null model is foundational to IMPULSE. It must be sufficiently accurate that real debris-induced impulses are detectable as statistically significant deviations. Of course, it cannot be perfect, some residual error is inevitable and must be characterized rather than eliminated.

The model propagates spacecraft mean elements under the dominant environmental perturbations.   	
\begin{itemize}
    \item Earth's gravity: $J_2$ through $J_4$ zonal harmonics. Higher-degree terms are negligible at the accuracy level of TLE-derived states.
    \item Atmospheric drag: Using a NRLMSISE-00-fit atmospheric density model with time-varying $F_{10.7}$ solar flux and $K_p$ geomagnetic indices. The ballistic coefficient $C_d A/m$ is treated as a time-varying parameter (see below).
    \item Solar radiation pressure: A cannonball model with cylindrical Earth shadowing. The spacecraft area-to-mass ratio is assumed known to within a factor of two.
    \item Third-body perturbations: Lunisolar gravitational effects.
\end{itemize}

Propagation uses orbit-averaged secular rates instead of osculating numerical integration. Brouwer theory provides $J_2$ precession rates for node and perigee. Drag and SRP rates are averaged over one revolution. This is a simplification, the rates vary slowly compared to the typical LEO 100-minute orbital period, and because the filter and truth propagation share identical physics, any systematic bias cancels in the residual. The residuals therefore represent unmodeled acceleration. 

$C_d$ is not treated as a constant. It follows a Ornstein-Uhlenbeck process with time constant $\tau_{C_d}$ and step variance chosen to match observed $C_d$ variability. This is meant to mirror the slow changes in attitude, surface properties and atmospheric composition without needing explicit attitude modeling. 

\subsubsection{State Representation}

The Gauss planetary equations for argument of perigee rate $d\omega/dt$ in orbital filtering poses a $1/e$ singularity. As eccentricity approaches zero, which it does for many objects in this regime due to drag-driven circularization, this term diverges. Numerical experiments confirmed that at $e=10^{-5}$, the argp-rate coefficient was already $10^7$ times larger than at $e=0.1$, causing filter instability.

To eliminate this singularity, the filter employs a non-singular state vector:
\begin{equation}
\mathbf{x} = [a,h,k,i,\Omega,M,w_R,w_S,w_W]^T,
\end{equation}
where $h=e\sin\omega$ and $k=e\cos\omega$ are the equinoctial-style substitutions. The chain-rule derivatives $dh/dt$ and $dk/dt$ cause the $1/e$ term to cancel algebraically against the $e$ implicit in $h$ and $k$ themselves. The measurement function in the first six states are able to be read out directly avoiding Jacobian singularities in this space. The measurement noise covariance is transformed from classical to $(h,k)$ space via the Jacobian of the transformation.


The unmodeled acceleration in the radial, along-track, and cross-track directions is represented by $[w_R,w_S,w_W]$. These are the direct observables for debris-induced impulses.

\subsubsection{EKF--Smoother Architecture}

The state is estimated with a forward Extended Kalman Filter( EKF) with a backward EKF and Fraser-Potter smoother. The reason behind this architecture is because TLEs are not raw measurements, but fitted elements that have complex noise properties. A simple batch least-squares fit would ignore the temporal nature of TLE errors and the coupling between epochs. Given the dynamics and measurements, the EKF-smoother framework provides a solid estimate of the state history.

\textbf{Forward pass.} The EKF predicts using the augmented dynamics:
\begin{equation}
\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x},C_d,\mathrm{env}) + \mathbf{B}(\mathbf{x})\mathbf{w},
\end{equation}
where $\mathbf{f}$ comprises the deterministic secular rates and $\mathbf{B}$ is the $9\times 3$ Gauss-VOP sensitivity matrix mapping RSW acceleration into element rates. The unmodeled acceleration $\mathbf{w}$ follows a first-order Gauss--Markov (FOGM) process:
\begin{equation}
\dot{\mathbf{w}} = -\frac{\mathbf{w}}{\tau} + \boldsymbol{\nu},\qquad \langle\boldsymbol{\nu}(t)\boldsymbol{\nu}(t')^T\rangle = q\delta(t-t')\mathbf{I}.
\end{equation}

The FOGM model allows the filter to track slow varying unmodeled accelerations like form the drag model, while remaining sensitive to impulsive events. The time constant $\tau=3\times 10^4$\,s and process noise strength $q=10^{-19}$ are tuning parameters that control the trade-off between smoothness and responsiveness. They are selected via the validation ladder (\S3.1).

The state transition matrix $\boldsymbol{\Phi}$ and process noise covariance $\mathbf{S}$ are computed via finite-difference linearization of the augmented dynamics and the discrete FOGM covariance, respectively. Prediction is sub-stepped at 1-hour intervals to prevent covariance explosion from the $\mathbf{B}$-matrix coupling over large steps.

\textbf{Measurement update.} TLE-derived classical elements are transformed to $[a,h,k,i,\Omega,M]$ and updated with measurement noise covariance $\mathbf{R}$ transformed accordingly. Angle wrapping is applied to $\Omega$ and $M$ to maintain linearity in the update.

\textbf{Backward pass.} A backward EKF is initialized with an inflated covariance ($100\times$) from the final forward state. It runs in reverse time, with the FOGM time constant negated. The inflated covariance is to reflect that the backward filter has no prior information at the final epoch.

\textbf{Smoother.} The Fraser--Potter smoother fuses forward a posteriori and backward a priori estimates at each epoch:
\begin{equation}
\mathbf{x}_S = \mathbf{W}\mathbf{x}_F + (\mathbf{I}-\mathbf{W})\mathbf{x}_B,\qquad \mathbf{W}=\mathbf{P}_B(\mathbf{P}_F+\mathbf{P}_B)^{-1}.
\end{equation}

The smoothed covariance $\mathbf{P}_S$ is computed from the Joseph form to ensure symmetry and positive definiteness. The smoothed estimate provides the unbiased estimate of the state and unmodded acceleration history, incorporating all measurements in the window. The smoothed acceleration is used for detection, not the forward-filter estimate. The smoother’s retrospective correction mitigates the noise that would dominate the forward filter estimate.

To work through the sever ill-conditioned characteristic of orbital covariance matrices (with condition numbers frequently reaching $\mathcal{O}(10^{16})$), the smoother uses a regularized symmetric eigendecomposition, falling back to a singular value decomposition (SVD) with additive diagonal jitter in case of numerical instability.

\subsubsection{Analysis Windowing and Edge Trimming}

A forward backward pass over a long duration multi-year TLE history is unstable for two reasons. Firstly, the filter drift accumulates over long arcs, meaning small errors in the drag model or Cd estimation compound. This causes the filter to diverge from the true trajectory. Second, and more critically, a debris strike early in the history corrupts the entire backward pass for later epochs. The backward pass, which starts from the final forward state, propagates the early epoch error backward in time, which affects the independence of later epochs.

To combat this, each spacecraft’s history is divided into independent analysis windows of $T_{\mathrm{win}}=90$ days. This duration is a compromise between two factors. Shorter windows reduce the accumulation of filter drift and confine backward pass corruption to a small temporal region, but they also reduce the number of measurements per window and increase the edge trimming penalty

\textbf{Window construction.} Windows are segments spanning the available TLE record, in this case, of 90 days. They are non-overlapping and treated as an independent statistical realization. The choice of non-overlapping windows is to ensure that detections are statistically independent between adjacent windows to simplify false-positive background subtraction and accounting.


\textbf{Edge trimming.} The EKF initialization transient and smoother boundary effects make the first and lasty $\DeltaT_{\mathrm{trim}}$ of each window unreliable. The filter is initialized with a generic prior on the unmodeled acceleration ($\mathbf{w}=\mathbf{0}$ with small covariance), and it takes time for the filter to converge to converge to a true dynamical state. At the end of the window, the inflated covariance of the backward window creates a boundary layer where the smoothed estimate is dominated by the forward filter only.

The trim margin is set to be $\max(7\,\mathrm{days},3\tau)$, where $\tau$ is the FOGM time constant. This ensures that the FOGM transient has decayed to less than 5\% of its initial value before data is used for detection. For $\tau=3\times 10^4$\,s ($\sim 8.3$ hours), this yields a trim margin of 7 days. The usable fraction per window is ~84.4\%, meaning that a 90 day window gives approximately 76 days of detection time.

\textbf{Statistical treatment.} Each trimmed window is one independent test. The aggregate search volume, with 15,830 total valid windows across the fleet, is substantial. This is the denominator of the false-positive rate calculation and the sample size of detection efficiency characterization.