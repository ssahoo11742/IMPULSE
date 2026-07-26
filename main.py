from TLE.tle_io import load_tles_from_file, parse_tle, tle_to_elements
from propagator.propagator import propagate_clean, propagate_debris
import numpy as np

tles = load_tles_from_file("TLE/real_tles.tle")
name, l1, l2 = tles[0]
t = parse_tle(name, l1, l2)
el0 = tle_to_elements(t)
rng_clean = np.random.default_rng(42)
rng_debris = np.random.default_rng(432)

result_clean = propagate_clean(el0, t["epoch_jd"], 30*86400, 86400, area=1.0, mass=100.0, Cd_base=2.2, rng=rng_clean)
result_debris, n_impacts = propagate_debris(el0, t["epoch_jd"], 30*86400, 86400, area=1.0, mass=100.0, rho_debris=1e-9, Cd_base=2.2, rng=rng_debris)

print(result_clean.final_elements)
print(result_debris.final_elements)

# Extract the final elements
c_el = result_clean.final_elements
d_el = result_debris.final_elements

print("\n=== ORBITAL DRIFT DIFF (CLEAN vs DEBRIS) ===")
print(f"Semi-major Axis (a) Diff : {c_el.a - d_el.a:+.3f} meters")
print(f"Eccentricity (ecc) Diff   : {c_el.ecc - d_el.ecc:+.8f}")
print(f"Inclination (inc) Diff   : {c_el.inc - d_el.inc:+.8f} rad")
print(f"RAAN Diff                : {c_el.raan - d_el.raan:+.8f} rad")
print(f"Arg of Perigee (argp) Diff: {c_el.argp - d_el.argp:+.8f} rad")
print(f"Mean Anomaly (M) Diff    : {c_el.M - d_el.M:+.8f} rad")
print(f"Total Debris Impacts     : {n_impacts}")