"""
Quick diagnostic: find out what's actually throwing exceptions at the
worst cell (730 days, rho=1e-8), and check whether reentry rates climb
with rho the way we'd expect.

IMPORTANT: this tries to use YOUR REAL phase1_config.py bounds/sampling
first. Testing this myself with guessed area/mass/altitude/inclination
ranges produced ZERO failures in ~70 trials at the worst cell - so either
the eccentricity-clamp hypothesis is wrong, or (more likely) the guessed
ranges below don't match your actual config closely enough to trigger it.
Your real run failed ~19% of the time overall, so it should NOT be rare -
if this script doesn't reproduce failures using your real config either,
that's a real signal the hypothesis needs rethinking, not just a sampling
fluke.

Run this directly:

    python diagnose_failures.py
"""

import math
import traceback
import numpy as np

from propagator.orbital import MeanElements
from phase1.phase1_runner import propagate_clean, propagate_debris

DURATION_DAYS = 730
N_TRIALS = 30  # small - just enough to catch a few failures

try:
    from phase1.phase1_config import PARAMETER_BOUNDS, sample_sobol, map_to_bounds, get_parameter_dict, PARAMETER_NAMES
    USING_REAL_CONFIG = True
    print("Using REAL phase1_config.py bounds - good, this should match your production run.\n")
except ImportError:
    USING_REAL_CONFIG = False
    print("WARNING: could not import phase1_config - falling back to GUESSED ranges.")
    print("If this script fails to reproduce anything, that's likely why - fix the")
    print("import path below or run this from inside your actual phase1/ package.\n")
    AREA_RANGE = (0.5, 10.0)
    MASS_RANGE = (10.0, 1000.0)
    ALT_RANGE_KM = (400.0, 1000.0)
    INC_RANGE_DEG = (0.0, 100.0)


def random_params(rng, seed):
    if USING_REAL_CONFIG:
        unit = sample_sobol(1, seed=seed)
        phys = map_to_bounds(unit)
        p = get_parameter_dict(phys[0])
        # IMPORTANT: use ALL sampled params, not just area/mass/alt/inc -
        # Cd_base and f107_base were previously hardcoded at nominal values
        # (2.2, 150.0), which may be exactly why no failures reproduced -
        # if the real failures need Cd/F10.7 near the edges of their range,
        # nominal values would never trigger them.
        return (p["area_m2"], p["mass_kg"], p["altitude_km"], p["inclination_deg"],
                p["Cd_base"], p["f107_base"])
    else:
        area = rng.uniform(*AREA_RANGE)
        mass = rng.uniform(*MASS_RANGE)
        alt_km = rng.uniform(*ALT_RANGE_KM)
        inc_deg = rng.uniform(*INC_RANGE_DEG)
        return area, mass, alt_km, inc_deg, 2.2, 150.0


def make_el0(alt_km, inc_deg):
    RE = 6371e3
    a0 = (alt_km * 1000 + RE) / (1 - 0.001)
    return MeanElements(a=a0, ecc=0.001, inc=math.radians(inc_deg), raan=0.0, argp=0.0, M=0.0)


def run_one(rho, seed):
    rng = np.random.default_rng(seed)
    area, mass, alt_km, inc_deg, Cd_base, f107_base = random_params(rng, seed)

    if area / mass > 0.05:
        return "rejected", None

    el0 = make_el0(alt_km, inc_deg)
    rng_c = np.random.default_rng(seed)
    rng_d = np.random.default_rng(seed)
    duration_s = DURATION_DAYS * 86400.0

    try:
        res_c = propagate_clean(el0, 2460000.5, duration_s, 86400.0, area, mass,
                                 f107_base=f107_base, Cd_base=Cd_base, rng=rng_c)
        res_d, n_imp = propagate_debris(el0, 2460000.5, duration_s, 86400.0, area, mass, rho,
                                         f107_base=f107_base, Cd_base=Cd_base, rng=rng_d)
        return "ok", (res_c.reentered, res_d.reentered)
    except Exception:
        print(f"\n--- FAILED (rho={rho:.0e}, area={area:.2f}, mass={mass:.1f}, "
              f"alt={alt_km:.0f}km, inc={inc_deg:.1f}deg, Cd={Cd_base:.2f}, "
              f"f107={f107_base:.0f}, seed={seed}) ---")
        traceback.print_exc()
        return "failed", None


def main():
    # NOTE ON RUNTIME: at 730 days + high area/mass, a single clean+debris
    # pair can take 1-5 seconds (much slower than the ~40ms/year seen at
    # smaller area/shorter duration earlier in this project) - each extra
    # debris impact costs a Kepler solve + Gauss-VOP update. N_TRIALS=30
    # here should take roughly 1-2 minutes; raise it if you need to catch
    # a rarer failure, but expect it to scale roughly linearly.
    print(f"=== Worst cell: duration={DURATION_DAYS}d, rho=1e-8, n_trials={N_TRIALS} ===\n")

    counts = {"ok": 0, "failed": 0, "rejected": 0}
    reentered_clean_count = 0
    reentered_debris_count = 0

    for i in range(N_TRIALS):
        status, result = run_one(rho=1e-8, seed=i)
        counts[status] += 1
        if status == "ok":
            reentered_clean, reentered_debris = result
            reentered_clean_count += int(reentered_clean)
            reentered_debris_count += int(reentered_debris)

    print(f"\n=== Summary at rho=1e-8 ===")
    print(counts)
    print(f"reentered_clean:  {reentered_clean_count}/{counts['ok']}")
    print(f"reentered_debris: {reentered_debris_count}/{counts['ok']}")

    print(f"\n=== Now checking whether reentry rate actually climbs with rho ===")
    print("(using fewer trials per rho here just to keep total runtime sane)")
    N_SWEEP = max(10, N_TRIALS // 2)
    for rho in [1e-12, 1e-11, 1e-10, 1e-9, 1e-8]:
        ok, failed, rejected = 0, 0, 0
        reentered_debris_count = 0
        for i in range(N_SWEEP):
            status, result = run_one(rho=rho, seed=1000 + i)  # different seeds than above
            if status == "ok":
                ok += 1
                reentered_debris_count += int(result[1])
            elif status == "failed":
                failed += 1
            else:
                rejected += 1
        print(f"  rho={rho:.0e}:  ok={ok:3d}  failed={failed:3d}  rejected={rejected:3d}  "
              f"reentered_debris={reentered_debris_count}/{ok if ok else 1}")


if __name__ == "__main__":
    main()