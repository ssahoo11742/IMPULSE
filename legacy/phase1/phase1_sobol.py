"""Phase 1 — Sobol Variance Decomposition (FIXED)

Computes first-order (S1) and total (ST) Sobol indices using Saltelli's
extension of the Sobol sequence with Jansen (1999) estimators.

FIXES applied:
  - Removed abs() from observable means (was distorting variance decomposition)
  - Added unwrapped along-track observable option
  - Updated default ensemble size to 16 for stability
  - Added envelope-aware sampling option

This is the expensive but definitive tier of sensitivity analysis.
It tells you, for each observable:
  - S1_i: Fraction of variance explained by parameter i ALONE
  - ST_i: Fraction of variance explained by parameter i including ALL interactions
  - S1_rho / ST_rho: The identifiability of the debris density signal

Reference: Saltelli, A. et al. (2010). "Variance based sensitivity analysis
of model output. Design and estimator for the total sensitivity index."
Computer Physics Communications, 181(2), 259-270.
"""

import numpy as np
from scipy.stats import qmc
from .phase1_config import (
    PARAMETER_NAMES, PARAMETER_BOUNDS, N_PARAMS, 
    map_to_bounds, get_parameter_dict
)
from .phase1_runner import run_paired_ensemble
from .phase1_envelope import load_envelope, is_in_envelope


def sobol_indices(n_samples, param_names=None, bounds_dict=None,
                  n_ensemble_per_point=16, observable='along_track_m_unwrapped',
                  seed_A=42, seed_B=43,
                  envelope_path=None, require_in_envelope=False):
    """Compute first-order and total Sobol indices.

    Uses Saltelli's extension: generate two independent Sobol samples A and B,
    then for each parameter i, create mixed matrix C_i = B with column i from A.

    Jansen (1999) estimators:
        V_i  = mean(Y_B * (Y_Ci - Y_A))          → first-order numerator
        S1_i = V_i / V_total

        V_Ti = 0.5 * mean((Y_A - Y_Ci)^2)        → total-effect numerator  
        ST_i = V_Ti / V_total

    Parameters:
        n_samples: Base sample size (total evals = n_samples * (2*n_params + 2))
        param_names: Parameter names (default: all)
        bounds_dict: Parameter bounds (default: PARAMETER_BOUNDS)
        n_ensemble_per_point: Ensemble size per evaluation (default: 16)
        observable: Observable to analyze (default: 'along_track_m_unwrapped')
        seed_A, seed_B: Seeds for the two independent Sobol sequences
        envelope_path: Path to operating envelope JSON for constrained sampling
        require_in_envelope: If True, resample points outside envelope

    Returns:
        dict with keys 'S1', 'ST', 'V_total', 'Y_mean', 'Y_std', 'n_samples'
    """
    if param_names is None:
        param_names = PARAMETER_NAMES
    if bounds_dict is None:
        bounds_dict = PARAMETER_BOUNDS

    n_params = len(param_names)

    # Load envelope if requested
    envelope = None
    if envelope_path is not None:
        try:
            envelope = load_envelope(envelope_path)
        except FileNotFoundError:
            envelope = None
    elif require_in_envelope:
        try:
            envelope = load_envelope()
        except FileNotFoundError:
            envelope = None

    print(f"Generating Sobol samples: A (seed={seed_A}), B (seed={seed_B}), n={n_samples}...")
    sampler_A = qmc.Sobol(d=n_params, scramble=True, seed=seed_A)
    sampler_B = qmc.Sobol(d=n_params, scramble=True, seed=seed_B)

    A = sampler_A.random(n=n_samples)
    B = sampler_B.random(n=n_samples)

    # Envelope filtering: resample points outside envelope
    if require_in_envelope and envelope is not None:
        print("Applying envelope constraints to samples...")
        A = _constrain_to_envelope(A, bounds_dict, param_names, envelope)
        B = _constrain_to_envelope(B, bounds_dict, param_names, envelope)

    A_phys = map_to_bounds(A, bounds_dict, param_names)
    B_phys = map_to_bounds(B, bounds_dict, param_names)

    # Evaluate base samples
    print(f"Evaluating base samples A and B ({n_samples} each)...")
    Y_A = []
    Y_B = []
    n_rejected = 0
    for i in range(n_samples):
        _, meta_A = run_paired_ensemble(get_parameter_dict(A_phys[i], param_names), 
                                        n_ensemble=n_ensemble_per_point)
        _, meta_B = run_paired_ensemble(get_parameter_dict(B_phys[i], param_names),
                                        n_ensemble=n_ensemble_per_point)

        # Handle rejections
        if meta_A is None or meta_A.get('rejected') or 'obs_mean' not in meta_A:
            n_rejected += 1
            Y_A.append(0.0)
        else:
            # FIX: Removed abs() — use signed mean for proper variance decomposition
            Y_A.append(meta_A['obs_mean'][observable])

        if meta_B is None or meta_B.get('rejected') or 'obs_mean' not in meta_B:
            n_rejected += 1
            Y_B.append(0.0)
        else:
            Y_B.append(meta_B['obs_mean'][observable])

    Y_A = np.array(Y_A)
    Y_B = np.array(Y_B)

    if n_rejected > 0:
        print(f"  Warning: {n_rejected} rejected/failed samples set to 0.0")

    # Total variance over combined sample
    V_total = np.var(np.concatenate([Y_A, Y_B]), ddof=1)

    # Compute mixed matrices and indices
    S1 = {}
    ST = {}

    print(f"Evaluating mixed samples C_i ({n_samples} x {n_params} = {n_samples*n_params} evaluations)...")
    for i, name in enumerate(param_names):
        C_i = B.copy()
        C_i[:, i] = A[:, i]
        C_i_phys = map_to_bounds(C_i, bounds_dict, param_names)

        Y_C = []
        n_rejected_C = 0
        for j in range(n_samples):
            _, meta_C = run_paired_ensemble(get_parameter_dict(C_i_phys[j], param_names),
                                            n_ensemble=n_ensemble_per_point)
            if meta_C is None or meta_C.get('rejected') or 'obs_mean' not in meta_C:
                n_rejected_C += 1
                Y_C.append(0.0)
            else:
                # FIX: No abs()
                Y_C.append(meta_C['obs_mean'][observable])

        if n_rejected_C > 0:
            print(f"    {name}: {n_rejected_C} rejected/failed in C_i")

        Y_C = np.array(Y_C)

        # Jansen estimators
        V_i = np.mean(Y_B * (Y_C - Y_A))
        S1[name] = float(V_i / V_total) if V_total > 0 else 0.0

        V_Ti = 0.5 * np.mean((Y_A - Y_C)**2)
        ST[name] = float(V_Ti / V_total) if V_total > 0 else 0.0

        print(f"  [{i+1}/{n_params}] {name}: S1={S1[name]:.4f}, ST={ST[name]:.4f}")

    return {
        'S1': S1,
        'ST': ST,
        'V_total': float(V_total),
        'Y_mean': float(np.mean(Y_A)),
        'Y_std': float(np.std(Y_A)),
        'n_samples': n_samples,
        'observable': observable,
        'n_rejected': n_rejected,
    }


def _constrain_to_envelope(unit_samples, bounds_dict, param_names, envelope, max_attempts=50):
    """Resample points outside the envelope until they fall inside."""
    n_samples, n_params = unit_samples.shape
    constrained = unit_samples.copy()

    for i in range(n_samples):
        params = get_parameter_dict(
            map_to_bounds(constrained[i:i+1], bounds_dict, param_names)[0],
            param_names
        )
        if is_in_envelope(params, envelope):
            continue

        # Try resampling duration and rho
        for _ in range(max_attempts):
            test = constrained[i].copy()
            if 'duration_days' in param_names:
                idx = param_names.index('duration_days')
                test[idx] = np.random.rand()
            if 'log10_rho_debris' in param_names:
                idx = param_names.index('log10_rho_debris')
                test[idx] = np.random.rand()

            params_test = get_parameter_dict(
                map_to_bounds(test.reshape(1, -1), bounds_dict, param_names)[0],
                param_names
            )
            if is_in_envelope(params_test, envelope):
                constrained[i] = test
                break

    return constrained


def compute_identifiability(sobol_result, target_param='log10_rho_debris'):
    """Compute identifiability score for the target parameter (rho).

    Score = S1_rho / (sum of S1_nuisance + residual)

    Where residual = 1 - sum(S1) captures higher-order interactions + noise.

    Returns dict with score, status, and components.
    """
    S1 = sobol_result['S1']
    s1_rho = max(0.0, S1.get(target_param, 0.0))
    s1_nuisance = sum(max(0.0, v) for k, v in S1.items() if k != target_param)
    residual = max(0.0, 1.0 - sum(max(0.0, v) for v in S1.values()))

    denominator = s1_nuisance + residual + 1e-12
    score = s1_rho / denominator

    if score < 0.1:
        status = "below_detection_floor"
    elif score < 0.3:
        status = "weakly_identifiable"
    elif score < 0.5:
        status = "identifiable"
    else:
        status = "strongly_identifiable"

    return {
        'score': float(score),
        'status': status,
        'S1_rho': float(s1_rho),
        'S1_nuisance_sum': float(s1_nuisance),
        'residual': float(residual),
    }


def print_sobol_results(sobol_result):
    """Pretty-print Sobol indices and identifiability."""
    S1 = sobol_result['S1']
    ST = sobol_result['ST']

    print("\n" + "=" * 80)
    print(f"SOBOL VARIANCE DECOMPOSITION — Observable: {sobol_result['observable']}")
    print("=" * 80)
    print(f"{'Parameter':20s} | {'S1 (first-order)':>18s} | {'ST (total)':>18s} | {'ST - S1':>12s}")
    print("-" * 80)

    sorted_params = sorted(S1.keys(), key=lambda k: S1[k], reverse=True)
    for name in sorted_params:
        s1 = S1[name]
        st = ST[name]
        diff = st - s1
        print(f"{name:20s} | {s1:18.6f} | {st:18.6f} | {diff:12.6f}")

    print("-" * 80)
    print(f"Sum of S1: {sum(S1.values()):.6f}")
    print(f"Sum of ST: {sum(ST.values()):.6f}")
    if sobol_result.get('n_rejected'):
        print(f"Rejected samples: {sobol_result['n_rejected']}")

    # Identifiability
    ident = compute_identifiability(sobol_result)
    print("\n" + "=" * 80)
    print("IDENTIFIABILITY ASSESSMENT")
    print("=" * 80)
    print(f"Target parameter: log10_rho_debris")
    print(f"S1_rho:           {ident['S1_rho']:.6f}")
    print(f"S1_nuisance_sum:  {ident['S1_nuisance_sum']:.6f}")
    print(f"Residual:         {ident['residual']:.6f}")
    print(f"Score:            {ident['score']:.4f}")
    print(f"Status:           {ident['status'].upper()}")
    print("=" * 80)


if __name__ == '__main__':
    # RIGOROUS production run
    print("Running Sobol analysis — RIGOROUS SETTINGS")
    print("=" * 80)
    print("n_samples=1024, n_ensemble=16, observable=along_track_m_unwrapped")
    print("WARNING: This will take several hours. Run overnight.")
    print("=" * 80)

    result = sobol_indices(
        n_samples=1024, 
        n_ensemble_per_point=16, 
        observable='along_track_m_unwrapped',
        require_in_envelope=True
    )
    print_sobol_results(result)