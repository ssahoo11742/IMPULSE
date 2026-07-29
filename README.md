# IMPULSE
---

## 1. Project Overview

IMPULSE is an orbit propagation and inference system designed to detect the presence of sub-radar-trackable orbital debris (fragments below ~10 cm) by analyzing the accumulated perturbations on a satellite's mean orbital elements over months to years. The core scientific insight is that debris impacts produce tiny, stochastic velocity kicks that accumulate into a detectable drift in the satellite's orbit. By running twin simulations — one "clean" (no debris) and one "debris-forced" — from identical initial conditions and random seeds, all deterministic natural forces cancel in the difference, isolating the debris signal.

The system operates entirely in **mean orbital element space**, not Cartesian ECI coordinates. This choice is deliberate: Two-Line Elements (TLEs), the primary public orbital data source, are themselves fits in mean-element space (consumed by SGP4). Propagating secular rates directly keeps the simulation in the same representation as the observational data and avoids the computational cost of integrating Cartesian state at sub-minute intervals.

---

## 2. System Architecture

The codebase is organized into six modules:

| Module | Responsibility |
|--------|---------------|
| `constants.py` | Centralized physical constants, empirical fit coefficients, and numerical tolerances. |
| `ephemeris.py` | Low-precision analytical position vectors for the Sun and Moon in ECI coordinates. |
| `atmosphere.py` | Upper-atmosphere density model with solar (F10.7) and geomagnetic (Kp) drivers. |
| `orbital.py` | Core orbital mechanics: secular gravity rates, drag, SRP, third-body effects, Kepler's equation, and Gauss Variation of Parameters. |
| `debris_impact_model.py` | Stochastic debris characterization (NSBM), impact geometry, von Mises-Fisher direction sampling, and Poisson statistics. |
| `propagator.py` | Time-integration engine: Euler stepping with clean and debris-forced propagation modes. |

---

## 3. State Representation: Mean Orbital Elements

A satellite's state is represented by the six **Keplerian mean elements**:

| Symbol | Name | Unit | Description |
|--------|------|------|-------------|
| $a$ | Semi-major axis | m | Half the major axis of the orbital ellipse. Determines orbital period via $T = 2\pi\sqrt{a^3/\mu}$. |
| $e$ | Eccentricity | — | Shape of the ellipse ($0 \leq e &lt; 1$). |
| $i$ | Inclination | rad | Tilt of the orbital plane relative to Earth's equator. |
| $\Omega$ | RAAN | rad | Right Ascension of the Ascending Node: where the orbit crosses the equator going north, measured from the vernal equinox. |
| $\omega$ | Argument of perigee | rad | Angle from the ascending node to the point of closest approach (perigee). |
| $M$ | Mean anomaly | rad | A fictitious angle increasing linearly with time: $M = M_0 + n(t-t_0)$. Parameterizes mean position. |

**Mean vs. Osculating Elements**: Real orbits exhibit short-period oscillations (wiggles within a single orbit) due to Earth's oblateness and other forces. "Mean" elements average these out, leaving only the long-term (secular) trends and slow periodic variations. This is the natural language of TLEs and long-term propagation.

The `MeanElements` dataclass provides:
- `mean_motion()`: Computes $n = \sqrt{\mu/a^3}$ (rad/s), with a safety clamp at $a_{\min} = 10^6$ m to prevent division by zero during reentry.
- `alt_m()`: Computes mean altitude as $a - R_{\text{Earth}}$ (circular approximation).
- `copy()`: Deep copy for simulation branching.

---

## 4. Physical Constants (`constants.py`)

All physical constants are tagged with their source. The file is divided into two sections: original mission-critical constants and refactored constants that were previously "stranded" as magic numbers in the computation modules.

### 4.1 Fundamental Astronautical Constants

| Constant | Value | Source | Role |
|----------|-------|--------|------|
| `MU` | $3.986004418 \times 10^{14}$ m³/s² | WGS84 | Earth's gravitational parameter $\mu = GM$. Appears in every orbital mechanics formula. |
| `R_EARTH` | $6.371 \times 10^6$ m | — | Mean Earth radius. |
| `RE_EQ` | $6.378137 \times 10^6$ m | WGS84 | Equatorial radius. Used in $J_2$ perturbation formulas. |
| `OMEGA_EARTH` | $7.2921150 \times 10^{-5}$ rad/s | WGS84 | Earth's rotation rate. |
| `MU_MOON` | $4.902800066 \times 10^{12}$ m³/s² | Montenbruck & Gill 2000 | Moon's gravitational parameter. |
| `MU_SUN` | $1.327124400 \times 10^{20}$ m³/s² | Montenbruck & Gill 2000 | Sun's gravitational parameter. |
| `J2` | $1.08263 \times 10^{-3}$ | EGM96 | Earth's oblateness coefficient. Dominant gravitational perturbation in LEO. |
| `J3` | $-2.53265 \times 10^{-6}$ | EGM96 | Pear-shaped term (defined but unused in current propagation). |
| `J4` | $-1.61962 \times 10^{-6}$ | EGM96 | Higher-order oblateness. ~1000× smaller than $J_2$. |
| `P_SRP` | $4.56 \times 10^{-6}$ N/m² | IAU | Solar radiation pressure at 1 AU. |
| `AU` | $1.496 \times 10^{11}$ m | IAU | Astronomical unit in meters. |
| `OBLIQUITY` | $23.43929111^\circ$ | J2000 | Ecliptic obliquity: angle between Earth's equatorial plane and orbital plane. |

### 4.2 Time Constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `SECONDS_PER_DAY` | 86400.0 | Conversion factor. |
| `T_YEAR` | $365.25 \times 86400$ s | Julian year. |
| `T_SIDEREAL` | $365.25636 \times 86400$ s | Sidereal year (relative to fixed stars). |
| `T_CARRINGTON` | $27.2753 \times 86400$ s | Synodic Carrington solar rotation period. |

### 4.3 Drag and Reentry

| Constant | Value | Meaning |
|----------|-------|---------|
| `REENTRY_ALT` | 100 km | Kármán line. Propagation halts below this altitude. |
| `CD_MEAN` | 2.2 | Mean drag coefficient (Moe & Moe 2005). |
| `CD_SIGMA` | 0.2 | Standard deviation of $C_d$ initialization. |
| `CD_MIN` / `CD_MAX` | 1.8 / 2.8 | Hard bounds for $C_d$ clipping. |
| `CD_TAU_S` | $5 \times 86400$ s | Ornstein-Uhlenbeck correlation time for $C_d$ drift. |
| `CD_DRIFT_FRAC` | 0.20 | Fractional amplitude of $C_d$ random walk per correlation time. |

### 4.4 TLE Observation Noise

These represent the approximate measurement uncertainty in public TLE data, propagated through Gauss VOP at a reference 600 km / 60° orbit:

| Constant | Value | Element |
|----------|-------|---------|
| `SIGMA_A_M` | 541 m | Semi-major axis |
| `SIGMA_ECC` | $1.2 \times 10^{-5}$ | Eccentricity |
| `SIGMA_INC` | $1.5 \times 10^{-5}$ rad | Inclination |
| `SIGMA_RAAN` | $2.0 \times 10^{-4}$ rad | RAAN |
| `SIGMA_ARGP` | $3.0 \times 10^{-4}$ rad | Argument of perigee |
| `SIGMA_M` | $8.5 \times 10^{-4}$ rad | Mean anomaly |

### 4.5 Debris Population

`DEBRIS_INC_POP` is a normalized probability distribution of debris inclinations derived from Klinkrad (2006), Table 2.1. The raw counts cluster at:
- 28° (communications)
- 51.6° (ISS/Russian orbits)
- 65°, 74°, 82° (various LEO inclinations)
- 97°–98° (sun-synchronous, heavily populated)
- 150°, 180° (retrograde and polar)

### 4.6 Atmosphere Table

`ATM_TABLE_KM` is a piecewise-exponential fit to NRLMSISE-00, giving $(h_0, \rho_0, H)$ for 50 km bands from 200–1000 km, where density within a band follows:
$$\rho(h) = \rho_0 \exp\left(-\frac{h - h_0}{H}\right)$$

### 4.7 Refactored Empirical Coefficients

These were previously hardcoded in computation modules:

**Ephemeris (Meeus, Ch. 25)**:
- `J2000_JD` = 2451545.0
- Sun mean anomaly: `SUN_MA_J2000_DEG` = 357.529, rate = 0.98560028°/day
- Sun mean longitude: `SUN_ML_J2000_DEG` = 280.459, rate = 0.98564736°/day
- Sun equation-of-center coefficients: 1.915°, 0.020°
- Sun distance: mean 1.00014 AU, ecc term 0.01671 AU, small term 0.00014 AU
- Moon mean longitude: 218.316°, rate 13.176396°/day
- Moon mean anomaly: 134.963°, rate 13.064993°/day
- Moon argument of latitude: 93.272°, rate 13.229350°/day
- Moon longitude perturbation amplitude: 6.289°
- Moon latitude perturbation amplitude: 5.128°
- Moon mean distance: 385000.6 km, variation amplitude: 20905.4 km

**Atmosphere Model**:
- `F107_BASELINE` = 150.0 (solar flux baseline)
- `F107_ANNUAL_AMP` = 20.0, `F107_CARRINGTON_AMP` = 15.0
- Storm sampling: rate 40/year, peak $K_p \sim \mathcal{N}(5.5, 1.0)$, duration $\sim \mathcal{N}(1.5, 0.5)$ days, min 0.1 days
- `QUIET_KP` = 2.0
- `STORM_PROFILE_PEAK_FRAC` = 1/3 (triangular storm peak location)
- Density scaling: `F107_DENSITY_SCALE_DENOM` = 100.0, `DENSITY_SOLAR_MAX_MULT` = 3.0, `KP_DENSITY_EXP_COEF` = 0.32

**Numerical Parameters**:
- `MEAN_MOTION_A_MIN_M` = 1e6 (safety clamp)
- `KEPLER_TOL` = 1e-10, `KEPLER_MAX_ITER` = 50
- `KEPLER_E_GUESS_THRESHOLD` = 0.8 (initial E guess switch)
- `GAUSS_VOP_INC_EPS` = 1e-10, `GAUSS_VOP_ECC_EPS` = 1e-20
- `SRP_CR_MEAN` = 1.3 (mean reflectivity)
- `THIRDBODY_MIN_DIST_M` = 1e6 (safety check for Sun/Moon distance)

**Debris Model**:
- `LC_MIN_M` = 1.0e-3 (1 mm anchor for NSBM)
- `NSBM_POWER_LAW_EXP` = 1.71
- `NSBM_AM_REGIME_BOUNDARY_M` = 1.67e-3
- Area-to-mass coefficients: small regime (-0.3, -1.4), large regime (0.97, 1.149)
- `VMF_KAPPA_MIN` = 1e-6, `VMF_KAPPA_MAX` = 200.0, bisection iterations = 80
- `ROTATION_SINGULARITY_EPS` = 1e-12

---

## 5. Environmental Models

### 5.1 Ephemeris (`ephemeris.py`)

The Sun and Moon positions are needed for:
1. **Solar Radiation Pressure**: Direction and distance to the Sun.
2. **Third-Body Perturbations**: Direction and distance to the Moon and Sun.

The module uses **low-precision analytical formulas** from Meeus, *Astronomical Algorithms* (Ch. 25). Accuracy is arcminutes for the Sun and ~1° for the Moon. This is sufficient because third-body and SRP terms are already small secular corrections; a rough direction and distance is adequate.

#### Sun Position Algorithm

**Input**: Epoch Julian Date (`epoch_jd`) and elapsed simulation time in seconds (`elapsed_s`).

**Step 1**: Compute days since J2000.0:
$$d = (\text{epoch\_jd} - 2451545.0) + \frac{\text{elapsed\_s}}{86400}$$

**Step 2**: Mean anomaly and mean longitude:
$$g = (357.529 + 0.98560028 \cdot d) \mod 360^\circ$$
$$L = (280.459 + 0.98564736 \cdot d) \mod 360^\circ$$

The coefficient $0.9856^\circ$/day is $360^\circ / 365.25$ days — one full ecliptic circle per year.

**Step 3**: Ecliptic longitude (equation of the center):
$$\lambda = L + 1.915^\circ \sin g + 0.020^\circ \sin 2g$$

This corrects for Earth's elliptical orbit. The $\sin g$ term is the first-order Fourier component of the eccentricity correction.

**Step 4**: Distance in AU:
$$r_{\text{AU}} = 1.00014 - 0.01671\cos g - 0.00014\cos 2g$$

Earth's orbital eccentricity is ~0.0167, so distance varies by ±1.67%.

**Step 5**: Ecliptic-to-Equatorial rotation:
$$\begin{aligned}
x &= r \cos\lambda \\
y &= r \sin\lambda \cos\varepsilon \\
z &= r \sin\lambda \sin\varepsilon
\end{aligned}$$

where $\varepsilon$ is the obliquity of the ecliptic (~23.44°). This rotates from the Sun's orbital plane (ecliptic) to Earth's equatorial frame (ECI).

#### Moon Position Algorithm

The Moon is more complex because its orbit is:
- Faster: ~13°/day (vs. the Sun's ~1°/day)
- More eccentric: $e \approx 0.055$
- Inclined: ~5.1° to the ecliptic
- Heavily perturbed by the Sun and Earth's oblateness

The code uses a **mean-orbit approximation** with three angles:

| Angle | Rate (°/day) | Period | Meaning |
|-------|-------------|--------|---------|
| $L$ | 13.176396 | 27.32 days (sidereal month) | Mean longitude |
| $M_{\text{moon}}$ | 13.064993 | 27.55 days (anomalistic month) | Mean anomaly |
| $F$ | 13.229350 | 27.21 days (draconic month) | Argument of latitude |

The rates differ because the perigee and node precess. The code applies one perturbation to longitude and one to latitude:

$$\lambda = L + 6.289^\circ \sin M_{\text{moon}}$$
$$\beta = 5.128^\circ \sin F$$
$$r = 385000.6\text{ km} - 20905.4\text{ km} \cdot \cos M_{\text{moon}}$$

Then applies the same ecliptic-to-equatorial rotation as the Sun.

---

### 5.2 Atmospheric Density (`atmosphere.py`)

Atmospheric drag is the dominant non-gravitational force in LEO. Density varies by orders of magnitude with altitude, solar EUV flux, and geomagnetic activity.

#### Base Density: Piecewise Exponential

The atmosphere is modeled as a stack of exponential layers from the `ATM_TABLE_KM` table. Within each 50 km band:
$$\rho(h) = \rho_0 \exp\left(-\frac{h - h_0}{H}\right)$$

where $H$ is the **scale height** — the altitude change over which density drops by a factor of $e$. Scale height increases with altitude because gravity's compressive effect weakens.

#### Solar Flux Variation (F10.7)

The F10.7 index (solar radio flux at 10.7 cm) proxies for EUV heating of the thermosphere. It is modeled as two superimposed sinusoids with fixed random phases (sampled once per simulation):

$$F_{10.7}(t) = F_{\text{base}} + A_{\text{annual}}\sin\left(\frac{2\pi t}{T_{\text{year}}} + \phi_1\right) + A_{\text{Carrington}}\sin\left(\frac{2\pi t}{T_{\text{Carrington}}} + \phi_2\right)$$

- Annual cycle amplitude: 20 sfu (solar cycle proxy)
- Carrington (~27 day) amplitude: 15 sfu (sunspot rotation)
- Baseline: 150 sfu

The phases are fixed per run because re-randomizing each timestep would turn a smooth periodic signal into uncorrelated noise.

#### Geomagnetic Storms (Kp)

Storms are modeled as a **Poisson process** with rate 40/year. Each storm has:
- Random onset time (uniform over simulation)
- Peak $K_p \sim \mathcal{N}(5.5, 1.0)$, clipped to [0, 9]
- Duration $\sim \mathcal{N}(1.5, 0.5)$ days, minimum 0.1 days

The storm profile is triangular, peaking at 1/3 of its duration:

$$K_p(t) = 2 + (\text{peak}_{K_p} - 2) \cdot \text{shape}(t)$$

where `shape` rises linearly to 1.0 at the 1/3 point and decays linearly thereafter.

#### Full Density Model

$$\rho_{\text{full}} = \rho_{\text{base}} \times \exp\left(\frac{F_{10.7} - 150}{100}\ln 3\right) \times \exp\left(0.32(K_p - 2)\right)$$

- **Solar term**: ~3× variation from solar min to max.
- **Geomagnetic term**: At $K_p = 7$, density is $e^{1.6} \approx 5\times$ the quiet background.

---

## 6. Force Models & Perturbation Theory (`orbital.py`)

The module computes time derivatives $\dot{a}, \dot{e}, \dot{i}, \dot{\Omega}, \dot{\omega}, \dot{M}$ for each force. These are **secular/orbit-averaged rates**: they represent the slow drift of the mean elements, not the rapid oscillations within a single orbit.

### 6.1 Earth's Non-Spherical Gravity: Brouwer Secular Rates ($J_2$ and $J_4$)

Earth's equatorial bulge ($J_2$) creates torques on the orbital plane and twists the ellipse within it. The rates are derived from perturbation theory (Vallado; Kozai 1959).

First, define auxiliary quantities:
$$n = \sqrt{\frac{\mu}{a^3}}, \quad p = a(1-e^2), \quad \eta = \sqrt{1-e^2}$$

**$J_2$ Terms (verified)**:

$$\frac{d\Omega}{dt} = -\frac{3}{2} n J_2 \left(\frac{R_e}{p}\right)^2 \cos i$$

$$\frac{d\omega}{dt} = \frac{3}{4} n J_2 \left(\frac{R_e}{p}\right)^2 (4 - 5\sin^2 i)$$

$$\delta n = \frac{3}{2} n J_2 \left(\frac{R_e}{p}\right)^2 \eta \left(1 - \frac{3}{2}\sin^2 i\right)$$

Physical interpretations:
- **Nodal regression** ($d\Omega/dt$): The orbital plane precesses westward for prograde orbits ($i &lt; 90^\circ$), eastward for retrograde. At $i = 90^\circ$, it stops.
- **Apsidal rotation** ($d\omega/dt$): Perigee drifts around the orbit. At the **critical inclination** $i_c = \sin^{-1}\sqrt{4/5} \approx 63.4^\circ$, the term vanishes. This is why Molniya orbits use $63.4^\circ$ — they need a fixed perigee over the northern hemisphere.
- **Mean motion correction**: Slightly modifies the orbital period.

**$J_4$ Terms (provisional)**: Smaller corrections carried over from earlier implementation. $J_4$ is ~1000× smaller than $J_2$, so independent re-verification is low priority.

### 6.2 Atmospheric Drag

Drag removes orbital energy, causing the orbit to spiral inward. The secular rates (King-Hele 1987, Eqs. 2.16/2.17) are:

$$\frac{da}{dt} = -2a\beta\rho v_{\text{circ}}$$

$$\frac{de}{dt} = -\frac{1}{2}\beta\rho v_{\text{circ}} e$$

where $\beta = C_d A/m$ is the **ballistic coefficient** (m²/kg).

Physical intuition:
- $\dot{a}$ is always negative: the orbit shrinks.
- $\dot{e}$ is negative: drag is strongest at perigee (lowest altitude), so it removes velocity disproportionately there, circularizing the orbit.
- As $a$ decreases, the satellite enters denser air, accelerating the death spiral.

### 6.3 Solar Radiation Pressure (SRP)

Photons carry momentum. Sunlight pushes the satellite outward when sunlit; in Earth's shadow, no push. This asymmetry pumps eccentricity over the long term.

The orbit-averaged rate (Montenbruck & Gill, Eq. 3.80) is:

$$\frac{de}{dt} = \frac{3}{2} \frac{a_{\text{srp}}}{na} \cdot f_{\text{xy}}$$

where:
- $a_{\text{srp}} = P_{\text{srp}} C_r \frac{A}{m} \left(\frac{1\text{ AU}}{r_{\text{sun}}}\right)^2$ is the instantaneous acceleration
- $C_r = 1.3$ is the mean reflectivity
- $f_{\text{xy}}$ is the fraction of the Sun vector projected into the orbital plane

### 6.4 Third-Body Perturbations (Sun and Moon)

For high orbits, lunisolar gravity dominates. For LEO, it is a weak secular effect. The Kozai-type rates (Montenbruck & Gill, Eqs. 3.91–3.93) are:

For each body (Moon, Sun):
$$\frac{di}{dt} = \frac{15}{8} \frac{n_{\text{body}}^2}{n} \alpha^3 e \sin 2i$$

$$\frac{de}{dt} = \frac{15}{16} \frac{n_{\text{body}}^2}{n} \alpha^3 \sqrt{1-e^2} \sin 2i$$

where $\alpha = a / r_{\text{body}}$ and $n_{\text{body}} = \sqrt{\mu_{\text{body}} / r_{\text{body}}^3}$.

The $\sin 2i$ dependence means these effects vanish at $i = 0^\circ$ and $90^\circ$, and peak at $45^\circ$.

---

## 7. Impact Mechanics

Debris impacts are treated as **instantaneous impulses** (momentum transfers too fast to integrate as continuous forces). Two mathematical bridges are required:

### 7.1 Kepler's Equation (`mean_to_true_anomaly`)

Gauss VOP requires the satellite's position at the moment of impact, parameterized by the **true anomaly** $\nu$ (the actual geometric angle from perigee). But the state stores **mean anomaly** $M$, which is related to the **eccentric anomaly** $E$ via Kepler's equation:

$$M = E - e\sin E$$

This is transcendental — no closed-form solution. Newton's method solves it:

$$E_{n+1} = E_n - \frac{E_n - e\sin E_n - M}{1 - e\cos E_n}$$

with initial guess $E_0 = M$ for $e &lt; 0.8$, or $E_0 = \pi$ for high-eccentricity orbits. Once $E$ converges (tolerance $10^{-10}$), true anomaly is:

$$\tan\frac{\nu}{2} = \sqrt{\frac{1+e}{1-e}} \tan\frac{E}{2}$$

### 7.2 Gauss Variation of Parameters (`gauss_vop`)

A debris impact delivers a velocity impulse $\Delta\vec{v}$ in the **RSW frame**:
- **R** (Radial): Earth-to-satellite direction
- **S** (Along-track): Perpendicular to R in the orbital plane, in the direction of motion
- **W** (Cross-track): Completes the right-handed frame (north orbital pole)

Given $(\Delta v_R, \Delta v_S, \Delta v_W)$ at true anomaly $\nu$, the instantaneous changes to the orbital elements are:

$$\Delta a = \frac{2a^2}{h}\left(e\sin\nu \cdot \Delta v_R + \frac{p}{r}\Delta v_S\right)$$

$$\Delta e = \frac{1}{h}\left(p\sin\nu \cdot \Delta v_R + ((p+r)\cos\nu + re)\Delta v_S\right)$$

$$\Delta i = \frac{r\cos(\omega+\nu)}{h}\Delta v_W$$

$$\Delta\Omega = \frac{r\sin(\omega+\nu)}{h\sin i}\Delta v_W$$

$$\Delta\omega = \frac{1}{he}\left(-p\cos\nu \cdot \Delta v_R + (p+r)\sin\nu \cdot \Delta v_S\right) - \frac{r\sin(\omega+\nu)\cos i}{h\sin i}\Delta v_W$$

where $p = a(1-e^2)$, $r = p/(1+e\cos\nu)$, and $h = \sqrt{\mu p}$.

Note: $a$ and $e$ change from radial and along-track impulses; $i$ and $\Omega$ change only from cross-track; $\omega$ changes from all three.

---

## 8. Debris Impact Model (`debris_impact_model.py`)

The debris model has three components: **when** (Poisson statistics), **what** (NSBM fragment distribution), and **which direction** (von Mises-Fisher).

### 8.1 Fragment Characterization (NSBM)

The NASA Standard Breakup Model describes the size distribution of fragments from on-orbit collisions.

**Characteristic Length** ($L_c$): Sampled from a power-law cumulative distribution:
$$N(&gt;L_c) \propto L_c^{-1.71}$$

Inverse CDF sampling:
$$L_c = L_{\min} \cdot (1 - u)^{-1/1.71}$$

where $u \sim \mathcal{U}(0,1)$ and $L_{\min} = 1$ mm.

**Area-to-Mass Ratio** ($A/m$): Piecewise power law in $\log_{10}$ space:

$$\log_{10}(A/m) = \begin{cases} -0.3\log_{10}L_c - 1.4 & L_c &lt; 1.67\text{ mm} \\ 0.97\log_{10}L_c + 1.149 & L_c \geq 1.67\text{ mm} \end{cases}$$

The boundary at 1.67 mm separates "fluffy" small fragments (paint, insulation) from denser structural pieces. **Crucially**, $L_c$ must be in meters for the logarithm, despite the conventional mm-based description of the regime boundary.

**Mass**: $m = A / (A/m) = \pi(L_c/2)^2 / (A/m)$

### 8.2 Impact Geometry & Relative Velocity

For two circular orbits at the same altitude but different inclinations $i_1$ and $i_2$:

$$v_{\text{rel}} = 2v_{\text{circ}}\left|\sin\left(\frac{\Delta i}{2}\right)\right|$$

This is the law of cosines for two velocity vectors of equal magnitude $v$ separated by angle $\Delta i$:
$$|\vec{v}_1 - \vec{v}_2| = \sqrt{v^2 + v^2 - 2v^2\cos\Delta i} = 2v\sin(\Delta i/2)$$

The angle $\theta$ of $\vec{v}_{\text{rel}}$ from the anti-velocity direction is:
$$\theta = \arctan2(\sin\Delta i, 1 - \cos\Delta i)$$

### 8.3 Impact Direction Distribution (von Mises-Fisher)

Debris impacts are not isotropic. Most catalogued debris is prograde, so collisions are biased toward **head-on** (anti-velocity direction). The impact direction is modeled as a **von Mises-Fisher (vMF)** distribution on the unit sphere — the spherical analog of a Gaussian.

The distribution is centered on $\vec{\mu} = (0, -1, 0)$ in RSW coordinates (anti-along-track) with concentration parameter $\kappa$.

**Computing $\kappa$ from the debris population**:

For each debris inclination group $(i_{\text{deg}}, \text{frac})$:
1. Compute $(v_{\text{rel}}, \theta)$ relative to the satellite's inclination.
2. Weight by **flux**: $w = \text{frac} \times v_{\text{rel}}$ (higher relative speed → more collisions).
3. Accumulate flux-weighted mean cosine: $\langle\cos\theta\rangle = \frac{\sum w\cos\theta}{\sum w}$

Then invert the **Langevin function**:
$$L(\kappa) = \coth\kappa - \frac{1}{\kappa} = \langle\cos\theta\rangle$$

This is the same function from paramagnetism in statistical mechanics. High $\kappa$ means tightly clustered head-on impacts; low $\kappa$ means nearly isotropic.

**Sampling from vMF**: Wood's (1994) rejection algorithm. The algorithm generates a random unit vector whose dot product with $\vec{\mu}$ is biased by $\kappa$.

**Rotation to RSW**: The sampled vector is generated in a local frame where the z-axis is $(0,0,1)$, then rotated so that the z-axis aligns with $\vec{\mu} = (0,-1,0)$ via an axis-angle rotation matrix (Rodrigues' formula).

### 8.4 Poisson Impact Process

The number of impacts in a timestep follows a Poisson distribution:
$$N \sim \text{Poisson}(\lambda), \quad \lambda = \rho \cdot A_{\text{sat}} \cdot \bar{v}_{\text{rel}} \cdot \Delta t$$

where:
- $\rho$ = debris number density (fragments/m³, referenced to $L_c \geq 1$ mm)
- $A_{\text{sat}}$ = satellite cross-sectional area
- $\bar{v}_{\text{rel}}$ = flux-weighted mean relative velocity
- $\Delta t$ = integration timestep

Dimensional analysis confirms: $[\text{m}^{-3}] \cdot [\text{m}^2] \cdot [\text{m/s}] \cdot [\text{s}] = [\text{dimensionless}]$.

For each impact:
1. Sample $L_c$ → compute fragment mass $m_f$
2. Sample $v_{\text{rel}}$ from the flux-weighted inclination distribution
3. Sample direction from vMF($\kappa$)
4. Compute $\Delta v = (m_f / M_{\text{sat}}) \cdot v_{\text{rel}}$ (momentum conservation)
5. Apply via Gauss VOP at the current true anomaly

---

## 9. Time Integration (`propagator.py`)

### 9.1 Euler Stepping

The propagator integrates the secular rates forward in time using **Euler's method**:

$$\vec{x}(t + \Delta t) = \vec{x}(t) + \dot{\vec{x}}(t) \cdot \Delta t$$

where $\vec{x} = (a, e, i, \Omega, \omega, M)$.

**Why Euler? Why not RK4?**
The rates are already **orbit-averaged/secular**. They vary slowly compared to the orbital period itself (minutes vs. hours). The environmental drivers (F10.7, Sun/Moon position) also change slowly compared to the 1-day timestep. Higher-order integration would buy negligible accuracy at significant computational cost. This is a deliberate simplification flagged for sensitivity analysis.

**Clamping and wrapping**:
- $e \geq 0$ is enforced with `max(0.0, e + \dot{e}\Delta t)`
- Angles are wrapped modulo $2\pi$

**Reentry check**: If altitude drops below 100 km, propagation halts.

### 9.2 Drag Coefficient Stochasticity (Ornstein-Uhlenbeck)

$C_d$ is not constant. Satellite attitude, surface degradation, and atmospheric composition variations cause it to drift. The model treats $C_d$ as an **Ornstein-Uhlenbeck process** — a mean-reverting random walk:

$$dC_d = \frac{1}{\tau}(C_{d,\text{base}} - C_d)dt + \sigma dW$$

Discretized (Euler-Maruyama):
$$C_d(t+\Delta t) = C_d(t) + \frac{1}{\tau}(C_{d,\text{base}} - C_d)\Delta t + \sigma_{\text{step}} \cdot \mathcal{N}(0,1)$$

where:
- $\tau = 5$ days (correlation time)
- $\sigma_{\text{step}} = 0.20 \cdot C_{d,\text{base}} \cdot \sqrt{2\Delta t / \tau}$

The process is clipped to $[1.8, 2.8]$ to prevent unphysical values.

---

## 10. Differential Propagation Strategy

The core detection methodology relies on **twin simulations**:

1. **`propagate_clean`**: All natural forces (gravity, drag, SRP, third-body). No debris.
2. **`propagate_debris`**: Identical setup, plus stochastic debris impacts at density $\rho$.

**Critical requirement**: Both runs must share:
- Identical initial orbital elements `el0`
- Identical epoch Julian Date `epoch_jd`
- Identical random number generator seed `rng`
- Identical duration, timestep, area, mass

Because every deterministic force (gravity, drag, SRP, lunisolar, atmosphere, Cd drift) is computed from the same state and same random draws in both runs, they cancel perfectly in the difference. The residual is:

$$\Delta\vec{x}_{\text{final}} = \vec{x}_{\text{debris}} - \vec{x}_{\text{clean}} = \sum_{\text{impacts}} \text{Gauss VOP kicks}$$

This differential signal is what DRIFTS attempts to invert: given observed orbital drift (from TLEs) and a clean propagation, infer the debris density $\rho$ that would produce the observed residual.

The detection floor is empirically estimated at $10^{-11}$ fragments/m³ for 500–700 km orbits over 1 year, though this is treated as a prior to be refined by sensitivity analysis.

---

## 11. Module Reference

### 11.1 `constants.py`
Central repository for all physical constants, empirical coefficients, and numerical parameters. See Section 4 for full listing. Key principle: any number that is (a) an empirical fit coefficient, (b) a repeated literal, or (c) a tunable numerical tolerance is named and documented here.

### 11.2 `ephemeris.py`
- `_days_since_j2000(epoch_jd, elapsed_s)`: Converts simulation time to days since J2000.0.
- `sun_position_eci(epoch_jd, elapsed_s)`: Returns Sun position vector in meters (ECI) using Meeus Ch. 25 low-precision formulas.
- `moon_position_eci(epoch_jd, elapsed_s)`: Returns Moon position vector in meters (ECI) using mean-orbit approximation with one perturbation term.

### 11.3 `atmosphere.py`
- `base_density(alt_m)`: Piecewise-exponential lookup from `ATM_TABLE_KM`.
- `sample_f107_phases(rng)`: Returns random phases for annual and Carrington sinusoids.
- `f107_at_time(t_s, phases, f_base)`: Computes F10.7 at time $t$.
- `sample_storm_events(duration_s, rng)`: Poisson-samples storm list.
- `kp_at_time(t_s, storm_events)`: Computes current $K_p$ from background + active storms.
- `density(alt_m, f107, kp)`: Full density with solar and geomagnetic multipliers.

### 11.4 `orbital.py`
- `MeanElements`: Dataclass for orbital state.
- `brouwer_rates(a, e, i)`: Returns dict of secular rates from $J_2$ and $J_4$.
- `drag_rates(a, e, Cd, A, m, rho)`: Returns $(\dot{a}, \dot{e})$ from atmospheric drag.
- `srp_ecc_rate(a, A, m, r_sun)`: Eccentricity pumping rate from solar radiation.
- `third_body_rates(a, e, i, r_moon, r_sun)`: Lunisolar rates on $(i, e)$.
- `mean_to_true_anomaly(M, e)`: Solves Kepler's equation via Newton's method.
- `gauss_vop(a, e, i, omega, nu, dv_rsw)`: Converts RSW impulse to element changes.

### 11.5 `debris_impact_model.py`
- `sample_fragment_lc(rng)`: Samples $L_c$ from NSBM power law.
- `fragment_area_to_mass(Lc)`: NSBM piecewise $A/m$ ratio.
- `fragment_mass(Lc)`: Mass from $L_c$ via $A/m$.
- `_rel_velocity_and_angle(sat_inc, debris_inc, v_circ)`: Geometry for inclined circular orbits.
- `compute_vmf_kappa(sat_inc)`: Flux-weighted vMF concentration from debris population.
- `_invert_langevin(mean_cos)`: Bisection solver for $L(\kappa) = \langle\cos\theta\rangle$.
- `sample_vmf_direction(kappa, rng)`: Wood's algorithm for vMF sampling on $S^2$.
- `sample_impact_velocity(sat_inc, v_circ, rng)`: Flux-weighted $v_{\text{rel}}$ sampler.
- `sample_impact_count(rho, A, v_rel_mean, dt, rng)`: Poisson impact count.
- `apply_impact(el, mass, sat_inc, v_circ, kappa, rng)`: Full impact event (sample + Gauss VOP application).

### 11.6 `propagator.py`
- `PropagationResult`: Dataclass for final state, history, and reentry flag.
- `_step_rates(el, Cd, A, m, epoch_jd, t, f107, kp)`: Aggregates all force rates for one step.
- `propagate_clean(...)`: Natural-force-only propagation.
- `propagate_debris(...)`: Natural forces + stochastic debris impacts.

---

## 12. Mathematical Appendices

### A. Derivation of Brouwer $J_2$ Secular Rates

The gravitational potential of an oblate Earth includes the $J_2$ term:
$$U_{J_2} = -\frac{\mu J_2 R_e^2}{2r^3}\left(3\sin^2\phi - 1\right)$$

where $\phi$ is geocentric latitude. Averaging this perturbation over one orbit and applying Lagrange's planetary equations yields the secular rates. For RAAN:

$$\frac{d\Omega}{dt} = -\frac{3nJ_2}{2}\left(\frac{R_e}{p}\right)^2\cos i$$

The $\cos i$ dependence arises because the equatorial bulge's torque on the orbit is proportional to the projection of the angular momentum vector onto the symmetry axis. For $\omega$, the $\sin^2 i$ term reflects how the bulge's gravitational gradient twists the ellipse within its plane.

### B. Relative Velocity for Inclined Circular Orbits

Two bodies in circular orbits at the same radius $r$ but different inclinations have velocity vectors:
$$\vec{v}_1 = v(\cos i_1, 0, \sin i_1) \quad \text{(in a frame where the node is along x)}$$
$$\vec{v}_2 = v(\cos i_2, 0, \sin i_2)$$

Wait — this assumes the nodes align. For the general case where only the inclination differs and RAAN is arbitrary, the relative speed depends on the relative inclination $\Delta i$ and the relative RAAN. However, for the **flux-weighted average** over all possible RAANs of the debris population, the effective relative speed simplifies to the formula used:

$$v_{\text{rel}} = 2v\sin(\Delta i / 2)$$

This is exact for co-altitude circular orbits when considering the magnitude of the velocity difference vector, independent of RAAN alignment, because the velocity vectors have equal magnitude and the angle between their orbital planes is $\Delta i$.

### C. Langevin Function and vMF Concentration

The von Mises-Fisher distribution on the unit sphere $S^2$ with concentration $\kappa$ and mean direction $\vec{\mu}$ has density:
$$f(\vec{x}) = \frac{\kappa}{4\pi\sinh\kappa}\exp(\kappa \vec{\mu}\cdot\vec{x})$$

The expected value of the projection onto the mean direction is:
$$\langle\cos\theta\rangle = \int_{S^2} (\vec{\mu}\cdot\vec{x}) f(\vec{x}) d\Omega = \coth\kappa - \frac{1}{\kappa} \equiv L(\kappa)$$

For small $\kappa$: $L(\kappa) \approx \kappa/3 - \kappa^3/45 + \dots$
For large $\kappa$: $L(\kappa) \approx 1 - 1/\kappa + 2e^{-2\kappa} + \dots$

Inverting this allows calibration of the direction distribution from the computed mean cosine of the debris flux geometry.
