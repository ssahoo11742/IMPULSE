"""Phase 1 — Morris Screening (Elementary Effects) — DEPRECATED

NOTE: This module is kept for backward compatibility, but the reviewer
recommends skipping Morris and running full Sobol directly since compute
cost is affordable. Morris was only meant as a cheap pre-filter.

If you still want to run it, use along_track_m_unwrapped for clean results.
"""

import numpy as np
from .phase1_config import PARAMETER_NAMES, PARAMETER_BOUNDS, N_PARAMS, map_to_bounds, get_parameter_dict
from .phase1_runner import run_paired_ensemble, REJECTION_AM_RATIO_MAX
from .phase1_envelope import load_envelope, is_in_envelope


def morris_screening(n_trajectories, param_names=None, bounds_dict=None,
                     n_levels=4, n_ensemble_per_point=4, observable='along_track_m_unwrapped',
                     envelope_path=None, max_resample_attempts=100,
                     require_in_envelope=True):
    """Run Morris Method for parameter screening.

    DEPRECATED: Consider using sobol_indices() directly for rigorous analysis.
    """
    if param_names is None:
        param_names = PARAMETER_NAMES
    if bounds_dict is None:
        bounds_dict = PARAMETER_BOUNDS

    envelope = None
    if envelope_path is not None:
        try:
            envelope = load_envelope(envelope_path)
        except FileNotFoundError:
            envelope = None
    else:
        try:
            envelope = load_envelope()
        except FileNotFoundError:
            envelope = None

    n_params = len(param_names)
    delta = 1.0 / (n_levels - 1)

    elementary_effects = {name: [] for name in param_names}
    envelope_stats = {'in_envelope': 0, 'out_of_envelope': 0, 'total_evals': 0,
                      'rejected': 0, 'failed': 0, 'resampled': 0}

    def _is_valid_params(params_dict):
        if require_in_envelope and envelope is not None:
            if not is_in_envelope(params_dict, envelope):
                return False, "out_of_envelope"
        area = params_dict.get("area_m2", 0)
        mass = params_dict.get("mass_kg", 1)
        if mass > 0 and area / mass > REJECTION_AM_RATIO_MAX:
            return False, "area_mass_ratio"
        return True, "ok"

    def _sample_valid_point(label="point"):
        for attempt in range(max_resample_attempts):
            x = np.random.rand(n_params)
            x = np.floor(x * (n_levels - 1)) / (n_levels - 1)
            x = np.clip(x, 0, 1 - delta)

            params = get_parameter_dict(
                map_to_bounds(x.reshape(1, -1), bounds_dict, param_names)[0],
                param_names
            )
            valid, reason = _is_valid_params(params)
            if valid:
                envelope_stats['in_envelope'] += 1
                return x

            if reason == "area_mass_ratio":
                x_test = x.copy()
                for _ in range(20):
                    if "area_m2" in param_names and "mass_kg" in param_names:
                        idx_area = param_names.index("area_m2")
                        idx_mass = param_names.index("mass_kg")
                        x_test[idx_area] = np.floor(np.random.rand() * (n_levels - 1)) / (n_levels - 1)
                        x_test[idx_mass] = np.floor(np.random.rand() * (n_levels - 1)) / (n_levels - 1)
                    params_test = get_parameter_dict(
                        map_to_bounds(x_test.reshape(1, -1), bounds_dict, param_names)[0],
                        param_names
                    )
                    valid2, _ = _is_valid_params(params_test)
                    if valid2:
                        envelope_stats['resampled'] += 1
                        envelope_stats['in_envelope'] += 1
                        return x_test

            elif reason == "out_of_envelope":
                x_test = x.copy()
                for _ in range(20):
                    if 'duration_days' in param_names:
                        idx_dur = param_names.index('duration_days')
                        x_test[idx_dur] = np.floor(np.random.rand() * (n_levels - 1)) / (n_levels - 1)
                    if 'log10_rho_debris' in param_names:
                        idx_rho = param_names.index('log10_rho_debris')
                        x_test[idx_rho] = np.floor(np.random.rand() * (n_levels - 1)) / (n_levels - 1)
                    params_test = get_parameter_dict(
                        map_to_bounds(x_test.reshape(1, -1), bounds_dict, param_names)[0],
                        param_names
                    )
                    valid2, reason2 = _is_valid_params(params_test)
                    if valid2:
                        envelope_stats['resampled'] += 1
                        envelope_stats['in_envelope'] += 1
                        return x_test
                    elif reason2 == "area_mass_ratio":
                        for _ in range(20):
                            if "area_m2" in param_names and "mass_kg" in param_names:
                                idx_area = param_names.index("area_m2")
                                idx_mass = param_names.index("mass_kg")
                                x_test[idx_area] = np.floor(np.random.rand() * (n_levels - 1)) / (n_levels - 1)
                                x_test[idx_mass] = np.floor(np.random.rand() * (n_levels - 1)) / (n_levels - 1)
                            params_test2 = get_parameter_dict(
                                map_to_bounds(x_test.reshape(1, -1), bounds_dict, param_names)[0],
                                param_names
                            )
                            valid3, _ = _is_valid_params(params_test2)
                            if valid3:
                                envelope_stats['resampled'] += 1
                                envelope_stats['in_envelope'] += 1
                                return x_test
                        break

        envelope_stats['out_of_envelope'] += 1
        print(f"Warning: Could not find valid {label} after {max_resample_attempts} attempts.")
        return x

    def _run_ensemble_safe(params_dict, n_ensemble):
        obs_array, meta = run_paired_ensemble(params_dict, n_ensemble=n_ensemble)
        if meta is not None and meta.get('rejected'):
            envelope_stats['rejected'] += 1
            return None, meta
        if meta is None or 'obs_mean' not in meta:
            envelope_stats['failed'] += 1
            return None, {"failed": True, "reason": "missing obs_mean in metadata"}
        return obs_array, meta

    for traj in range(n_trajectories):
        x_base = _sample_valid_point(label="base point")
        D = np.diag(np.random.choice([-1, 1], n_params))
        x_prev = x_base.copy()

        for i in range(n_params):
            x_new = x_prev.copy()
            x_new[i] += delta * D[i, i]
            x_new = np.clip(x_new, 0, 1)
            x_new = _sample_valid_point(label=f"step {i} point")

            diff_params = np.where(np.abs(x_new - x_prev) > 1e-12)[0]
            if len(diff_params) > 1 or (len(diff_params) == 1 and diff_params[0] != i):
                x_test = x_prev.copy()
                found_valid = False
                for step_attempt in range(max_resample_attempts):
                    x_test[i] = np.floor(np.random.rand() * (n_levels - 1)) / (n_levels - 1)
                    params_test = get_parameter_dict(
                        map_to_bounds(x_test.reshape(1, -1), bounds_dict, param_names)[0],
                        param_names
                    )
                    valid, _ = _is_valid_params(params_test)
                    if valid:
                        x_new = x_test.copy()
                        found_valid = True
                        break
                if not found_valid:
                    print(f"  Skipping trajectory {traj}, step {i}: cannot find valid step for parameter {param_names[i]}")
                    x_prev = x_new
                    continue

            params_prev = get_parameter_dict(
                map_to_bounds(x_prev.reshape(1, -1), bounds_dict, param_names)[0],
                param_names
            )
            params_new = get_parameter_dict(
                map_to_bounds(x_new.reshape(1, -1), bounds_dict, param_names)[0],
                param_names
            )

            obs_prev, meta_prev = _run_ensemble_safe(params_prev, n_ensemble_per_point)
            obs_new, meta_new = _run_ensemble_safe(params_new, n_ensemble_per_point)

            envelope_stats['total_evals'] += 2

            if obs_prev is None or obs_new is None:
                print(f"  Skipping trajectory {traj}, step {i}: prev_rejected={meta_prev.get('rejected', False)}, new_rejected={meta_new.get('rejected', False)}")
                x_prev = x_new
                continue

            y_prev = abs(meta_prev['obs_mean'][observable])
            y_new = abs(meta_new['obs_mean'][observable])

            low, high = bounds_dict[param_names[i]]
            param_range = high - low

            ee = (y_new - y_prev) / (delta * param_range)
            elementary_effects[param_names[i]].append(ee)

            x_prev = x_new

    results = {}
    for name in param_names:
        ees = np.array(elementary_effects[name])
        if len(ees) == 0:
            print(f"Warning: No valid elementary effects for parameter '{name}'.")
            results[name] = {
                'mu': float('nan'),
                'mu_star': float('nan'),
                'sigma': float('nan'),
                'n_samples': 0
            }
        else:
            results[name] = {
                'mu': float(np.mean(ees)),
                'mu_star': float(np.mean(np.abs(ees))),
                'sigma': float(np.std(ees)),
                'n_samples': len(ees)
            }

    results['_envelope_stats'] = envelope_stats
    return results


def classify_morris(morris_results):
    param_results = {k: v for k, v in morris_results.items() if not k.startswith('_')}
    valid_mu_stars = [r['mu_star'] for r in param_results.values() if not np.isnan(r['mu_star'])]
    if not valid_mu_stars:
        return {k: 'unimportant' for k in param_results}

    max_mu_star = max(valid_mu_stars)
    classifications = {}

    for name, r in param_results.items():
        if np.isnan(r['mu_star']):
            classifications[name] = 'unimportant'
        elif r['mu_star'] > 0.1 * max_mu_star:
            if r['sigma'] > r['mu_star']:
                classifications[name] = 'nonlinear_interactive'
            else:
                classifications[name] = 'linear_important'
        else:
            classifications[name] = 'unimportant'

    return classifications


def print_morris_results(morris_results):
    classifications = classify_morris(morris_results)
    sorted_params = sorted(
        [k for k in morris_results.keys() if not k.startswith('_')], 
        key=lambda k: morris_results[k]['mu_star'] if not np.isnan(morris_results[k]['mu_star']) else -1, 
        reverse=True
    )

    print("=" * 80)
    print("MORRIS SCREENING RESULTS (DEPRECATED — use Sobol for rigor)")
    print("=" * 80)
    print(f"{'Parameter':20s} | {'mu':>10s} | {'mu_star':>10s} | {'sigma':>10s} | {'n_samples':>9s} | {'Classification':20s}")
    print("-" * 80)

    for name in sorted_params:
        r = morris_results[name]
        cls = classifications[name]
        mu_str = f"{r['mu']:10.4f}" if not np.isnan(r['mu']) else "     N/A   "
        mu_star_str = f"{r['mu_star']:10.4f}" if not np.isnan(r['mu_star']) else "     N/A   "
        sigma_str = f"{r['sigma']:10.4f}" if not np.isnan(r['sigma']) else "     N/A   "
        print(f"{name:20s} | {mu_str} | {mu_star_str} | {sigma_str} | {r['n_samples']:9d} | {cls:20s}")

    print("-" * 80)

    if '_envelope_stats' in morris_results:
        stats = morris_results['_envelope_stats']
        print(f"\nEnvelope diagnostics:")
        print(f"  Total evaluations: {stats['total_evals']}")
        print(f"  In envelope: {stats['in_envelope']}")
        print(f"  Out of envelope: {stats['out_of_envelope']}")
        print(f"  Resampled: {stats['resampled']}")
        print(f"  Rejected: {stats['rejected']}")
        print(f"  Failed: {stats['failed']}")

    print("\nNOTE: Morris is approximate. Run sobol_indices() for definitive results.")


if __name__ == '__main__':
    print("WARNING: Morris screening is deprecated. Use Sobol for rigorous analysis.")
    print("Running minimal Morris test with unwrapped observable...")
    results = morris_screening(
        n_trajectories=8, 
        n_ensemble_per_point=2, 
        observable='along_track_m_unwrapped',
        require_in_envelope=True
    )
    print_morris_results(results)