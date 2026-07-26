import copy
import math
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from SALib.sample import morris as morris_sample
from SALib.analyze import morris as morris_analyze
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

# Adapt imports to your local directory structure
from propagator.propagator import propagate_clean, propagate_debris


def run_paired_trial(params_dict, el0, epoch_jd, duration_s, dt_s, seed):
    """
    Executes a single paired (Clean vs Debris) run with identical PRNG seeds
    to isolate the differential debris signal from deterministic/nuisance variations.
    Includes explicit guards against orbital decay and unphysical numerical states.
    """
    rho_debris = params_dict["rho_debris"]
    Cd_base = params_dict["Cd_base"]
    area = params_dict["area"]
    mass = params_dict["mass"]
    f107_base = params_dict["f107_base"]

    # Ensure separate but identical random streams
    rng_clean = np.random.default_rng(seed)
    rng_debris = np.random.default_rng(seed)

    el0_clean = copy.deepcopy(el0)
    el0_debris = copy.deepcopy(el0)

    try:
        # 1. Baseline propagation
        res_clean = propagate_clean(
            el0=el0_clean,
            epoch_jd=epoch_jd,
            duration_s=duration_s,
            dt_s=dt_s,
            area=area,
            mass=mass,
            f107_base=f107_base,
            Cd_base=Cd_base,
            rng=rng_clean,
            record_history=False,
        )

        # 2. Debris propagation
        res_debris, total_impacts = propagate_debris(
            el0=el0_debris,
            epoch_jd=epoch_jd,
            duration_s=duration_s,
            dt_s=dt_s,
            area=area,
            mass=mass,
            rho_debris=rho_debris,
            f107_base=f107_base,
            Cd_base=Cd_base,
            rng=rng_debris,
            record_history=False,
        )

        # Handle early termination / reentry
        if res_clean.reentered or res_debris.reentered:
            return {
                "delta_a": np.nan,
                "delta_ecc": np.nan,
                "delta_inc": np.nan,
                "reentered": True,
                "total_impacts": total_impacts if 'total_impacts' in locals() else 0,
            }

        # Validate semi-major axis safety bounds
        if res_clean.final_elements.a <= 0 or res_debris.final_elements.a <= 0:
            return {
                "delta_a": np.nan,
                "delta_ecc": np.nan,
                "delta_inc": np.nan,
                "reentered": True,
                "total_impacts": total_impacts,
            }

        # 3. Compute differential observables
        delta_a = res_debris.final_elements.a - res_clean.final_elements.a
        delta_ecc = res_debris.final_elements.ecc - res_clean.final_elements.ecc
        delta_inc = res_debris.final_elements.inc - res_clean.final_elements.inc

        return {
            "delta_a": delta_a,
            "delta_ecc": delta_ecc,
            "delta_inc": delta_inc,
            "reentered": False,
            "total_impacts": total_impacts,
        }

    except (ValueError, FloatingPointError, OverflowError, ZeroDivisionError) as err:
        # Fallback safeguard against unphysical math domain errors during simulation
        return {
            "delta_a": np.nan,
            "delta_ecc": np.nan,
            "delta_inc": np.nan,
            "reentered": True,
            "total_impacts": 0,
        }


def _worker_task(args):
    """Unpacks arguments for process pool workers."""
    param_row, param_names, el0, epoch_jd, duration_s, dt_s, base_seed, replicate_idx = args
    params_dict = dict(zip(param_names, param_row))
    
    # Generate deterministic trial seed
    trial_seed = base_seed + hash((tuple(param_row), replicate_idx)) % (2**31 - 1)
    
    return run_paired_trial(params_dict, el0, epoch_jd, duration_s, dt_s, trial_seed)


def run_morris_screening(
    problem,
    el0,
    epoch_jd,
    duration_s=86400 * 7,
    dt_s=86400,
    N=10,
    num_levels=4,
    num_workers=4,
    base_seed=42
):
    """
    Tier 1: Fast Morris Elementary Effects Screening.
    Evaluates parameter importance with N * (D + 1) evaluations.
    """
    print("=" * 60)
    print("RUNNING TIER 1: MORRIS ELEMENTARY EFFECTS SCREENING")
    print("=" * 60)

    param_values = morris_sample.sample(problem, N=N, num_levels=num_levels)
    num_design_points = param_values.shape[0]
    param_names = problem["names"]

    print(f"--> Generated {num_design_points} Morris design trajectories (N={N}).")

    tasks = [
        (param_values[i], param_names, el0, epoch_jd, duration_s, dt_s, base_seed, 0)
        for i in range(num_design_points)
    ]

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for res in executor.map(_worker_task, tasks, chunksize=16):
            results.append(res)

    y_delta_a = np.array([r["delta_a"] for r in results])

    # Filter invalid trajectories
    valid_mask = ~np.isnan(y_delta_a)
    valid_count = np.sum(valid_mask)
    print(f"--> Valid runs completed: {valid_count}/{num_design_points} ({valid_count/num_design_points*100:.1f}%)")

    if valid_count < (num_design_points * 0.5):
        print("WARNING: More than 50% of runs reentered or threw math errors. Adjust parameter bounds or duration.")

    # Substitute NaNs with 0 signal for analysis stabilization if necessary
    y_analyzable = np.nan_to_num(y_delta_a, nan=0.0)

    morris_res = morris_analyze.analyze(
        problem,
        param_values,
        y_analyzable,
        conf_level=0.95,
        print_to_console=False,
        num_levels=num_levels
    )

    print("\n--- MORRIS SCREENING RESULTS (Observable: Delta SMA) ---")
    for name, mu_star, sigma in zip(param_names, morris_res["mu_star"], morris_res["sigma"]):
        print(f"  * {name:<12}: mu* = {mu_star:12.6e} | sigma = {sigma:12.6e}")
    print("=" * 60 + "\n")

    return morris_res


def run_sobol_sensitivity_analysis(
    problem,
    el0,
    epoch_jd,
    duration_s=86400 * 30,
    dt_s=86400,
    n_samples=256,
    n_replicates=4,
    num_workers=4,
    base_seed=42
):
    """
    Tier 2: Sobol Variance Decomposition.
    Evaluates N * (2D + 2) design points with replicate noise filtering.
    """
    print("=" * 60)
    print("RUNNING TIER 2: SOBOL VARIANCE DECOMPOSITION")
    print("=" * 60)

    param_values = sobol_sample.sample(problem, n_samples, calc_second_order=True)
    num_design_points = param_values.shape[0]
    param_names = problem["names"]

    total_trials = num_design_points * n_replicates
    print(f"--> Generated {num_design_points} Sobol design points.")
    print(f"--> Total paired executions (n={n_replicates} replicates): {total_trials}")

    tasks = []
    for i in range(num_design_points):
        for rep in range(n_replicates):
            tasks.append((param_values[i], param_names, el0, epoch_jd, duration_s, dt_s, base_seed, rep))

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for res in executor.map(_worker_task, tasks, chunksize=16):
            results.append(res)

    delta_a_matrix = np.zeros((num_design_points, n_replicates))
    idx = 0
    for i in range(num_design_points):
        for rep in range(n_replicates):
            delta_a_matrix[i, rep] = results[idx]["delta_a"]
            idx += 1

    # Take the replicate mean for variance decomposition
    y_delta_a = np.nanmean(delta_a_matrix, axis=1)
    y_analyzable = np.nan_to_num(y_delta_a, nan=0.0)

    si_delta_a = sobol_analyze.analyze(
        problem,
        y_analyzable,
        calc_second_order=True,
        print_to_console=False
    )

    print("\n--- SOBOL INDICES (Observable: Delta SMA) ---")
    print(f"{'Parameter':<12} | {'S1 (Direct)':<12} | {'ST (Total)':<12}")
    print("-" * 42)
    for name, s1, st in zip(param_names, si_delta_a["S1"], si_delta_a["ST"]):
        print(f"{name:<12} | {s1:12.4f} | {st:12.4f}")
    print("=" * 60 + "\n")

    return si_delta_a

import numpy as np
from SALib.analyze import morris as morris_analyze
from SALib.analyze import sobol as sobol_analyze

# Observables we care about
OBSERVABLES = ["delta_a", "delta_ecc", "delta_inc"]

def run_morris_screening_multi(
    problem,
    el0,
    epoch_jd,
    duration_s=86400 * 7,
    dt_s=86400,
    N=10,
    num_levels=4,
    num_workers=4,
    base_seed=42
):
    """
    Multi-observable Morris Elementary Effects Screening.
    Evaluates parameter importance across Delta-a, Delta-e, and Delta-i.
    """
    print("=" * 70)
    print("RUNNING TIER 1: MULTI-OBSERVABLE MORRIS SCREENING")
    print("=" * 70)

    param_values = morris_sample.sample(problem, N=N, num_levels=num_levels)
    num_design_points = param_values.shape[0]
    param_names = problem["names"]

    tasks = [
        (param_values[i], param_names, el0, epoch_jd, duration_s, dt_s, base_seed, 0)
        for i in range(num_design_points)
    ]

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for res in executor.map(_worker_task, tasks, chunksize=16):
            results.append(res)

    morris_results_by_obs = {}

    for obs in OBSERVABLES:
        y_obs = np.array([r[obs] for r in results])
        y_analyzable = np.nan_to_num(y_obs, nan=0.0)

        res = morris_analyze.analyze(
            problem,
            param_values,
            y_analyzable,
            conf_level=0.95,
            print_to_console=False,
            num_levels=num_levels
        )
        morris_results_by_obs[obs] = res

        print(f"\n--- MORRIS RESULTS: {obs.upper()} ---")
        print(f"{'Parameter':<12} | {'mu*':<14} | {'sigma':<14}")
        print("-" * 46)
        for name, mu_star, sigma in zip(param_names, res["mu_star"], res["sigma"]):
            print(f"{name:<12} | {mu_star:14.6e} | {sigma:14.6e}")

    print("=" * 70 + "\n")
    return morris_results_by_obs


def run_sobol_sensitivity_analysis_multi(
    problem,
    el0,
    epoch_jd,
    duration_s=86400 * 14,
    dt_s=86400,
    n_samples=32,
    n_replicates=2,
    num_workers=4,
    base_seed=42
):
    """
    Multi-observable Sobol Variance Decomposition across all orbital parameters.
    """
    print("=" * 70)
    print("RUNNING TIER 2: MULTI-OBSERVABLE SOBOL DECOMPOSITION")
    print("=" * 70)

    param_values = sobol_sample.sample(problem, n_samples, calc_second_order=True)
    num_design_points = param_values.shape[0]
    param_names = problem["names"]

    tasks = []
    for i in range(num_design_points):
        for rep in range(n_replicates):
            tasks.append((param_values[i], param_names, el0, epoch_jd, duration_s, dt_s, base_seed, rep))

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for res in executor.map(_worker_task, tasks, chunksize=16):
            results.append(res)

    sobol_results_by_obs = {}

    for obs in OBSERVABLES:
        obs_matrix = np.zeros((num_design_points, n_replicates))
        idx = 0
        for i in range(num_design_points):
            for rep in range(n_replicates):
                obs_matrix[i, rep] = results[idx][obs]
                idx += 1

        y_obs = np.nanmean(obs_matrix, axis=1)
        y_analyzable = np.nan_to_num(y_obs, nan=0.0)

        si_res = sobol_analyze.analyze(
            problem,
            y_analyzable,
            calc_second_order=True,
            print_to_console=False
        )
        sobol_results_by_obs[obs] = si_res

        print(f"\n--- SOBOL INDICES: {obs.upper()} ---")
        print(f"{'Parameter':<12} | {'S1 (Direct)':<12} | {'ST (Total)':<12}")
        print("-" * 42)
        for name, s1, st in zip(param_names, si_res["S1"], si_res["ST"]):
            print(f"{name:<12} | {s1:12.4f} | {st:12.4f}")

    print("=" * 70 + "\n")
    return sobol_results_by_obs