"""Phase 1 — Signal Existence Audit: Main Orchestration

Top-level script that runs the complete Phase 1 analysis:
  1. pilot          — Quick validation (16 points)
  2. morris         — Morris screening only
  3. sobol          — Full Sobol analysis
  4. identifiability — Duration-rho operating envelope map
  5. full           — Complete pipeline

Usage:
    python phase1_main.py --mode pilot
    python phase1_main.py --mode morris --n 20
    python phase1_main.py --mode sobol --n 256
    python phase1_main.py --mode identifiability --n 128
    python phase1_main.py --mode full --n 512
"""

import argparse
import json
import pickle
import time
import numpy as np
from pathlib import Path

from .phase1_config import (
    PARAMETER_NAMES, PARAMETER_BOUNDS, N_PARAMS, N_ENSEMBLE,
    sample_sobol, sample_lhs, map_to_bounds, get_parameter_dict
)
from .phase1_runner import run_paired_ensemble
from .phase1_morris import morris_screening, print_morris_results
from .phase1_sobol import sobol_indices, print_sobol_results, compute_identifiability


def run_pilot(n_design=16, n_ensemble=4, output_dir="./phase1_results"):
    """Quick pilot study to validate the pipeline."""
    print("=" * 80)
    print("PHASE 1 PILOT STUDY")
    print("=" * 80)
    print(f"Design points: {n_design}")
    print(f"Ensemble size: {n_ensemble}")
    print(f"Total paired propagations: {n_design * n_ensemble}")
    
    unit_samples = sample_sobol(n_design, seed=42)
    phys_samples = map_to_bounds(unit_samples)
    
    results = []
    t0 = time.time()
    for idx in range(n_design):
        params = get_parameter_dict(phys_samples[idx])
        obs_array, meta = run_paired_ensemble(params, n_ensemble=n_ensemble)
        results.append({
            'design_idx': idx,
            'params': params,
            'obs_mean': meta['obs_mean'],
            'obs_std': meta['obs_std'],
            'obs_cv': meta['obs_cv'],
        })
        if (idx + 1) % 4 == 0 or idx == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            remaining = (n_design - idx - 1) / rate if rate > 0 else 0
            print(f"  [{idx+1}/{n_design}] {elapsed:.1f}s elapsed, ~{remaining:.0f}s remaining")
    
    total_time = time.time() - t0
    print(f"\nPilot complete in {total_time:.1f}s ({total_time/60:.1f} min)")
    
    obs_names = list(results[0]["obs_mean"].keys())
    print("\nObservable summary across design space:")
    for oname in ['delta_a_m', 'delta_ecc', 'along_track_m', 'total_impacts']:
        if oname in obs_names:
            values = [r["obs_mean"][oname] for r in results]
            print(f"  {oname:20s}: mean={np.mean(values):12.4e}, std={np.std(values):12.4e}, "
                  f"range=[{np.min(values):12.4e}, {np.max(values):12.4e}]")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f'{output_dir}/pilot_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved: {output_dir}/pilot_results.pkl")
    
    return results


def run_morris(n_trajectories=20, n_ensemble=4, output_dir="./phase1_results"):
    """Run Morris screening analysis."""
    print("\n" + "=" * 80)
    print("MORRIS SCREENING")
    print("=" * 80)
    print(f"Trajectories: {n_trajectories}")
    print(f"Steps per trajectory: {N_PARAMS}")
    print(f"Evaluations: {n_trajectories * N_PARAMS * 2}")
    print(f"Ensemble per evaluation: {n_ensemble}")
    print(f"Total propagations: {n_trajectories * N_PARAMS * 2 * n_ensemble}")
    
    t0 = time.time()
    results = morris_screening(n_trajectories=n_trajectories, 
                               n_ensemble_per_point=n_ensemble,
                               observable="along_track_m")
    elapsed = time.time() - t0
    print(f"\nMorris complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    
    print_morris_results(results)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f'{output_dir}/morris_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_dir}/morris_results.json")
    
    return results


def run_sobol(n_samples=256, n_ensemble=4, output_dir="./phase1_results"):
    """Run Sobol variance decomposition."""
    print("\n" + "=" * 80)
    print("SOBOL VARIANCE DECOMPOSITION")
    print("=" * 80)
    print(f"Base samples: {n_samples}")
    print(f"Total evaluations: {n_samples * (2*N_PARAMS + 2)}")
    print(f"Ensemble per evaluation: {n_ensemble}")
    print(f"Total propagations: {n_samples * (2*N_PARAMS + 2) * n_ensemble}")
    print("WARNING: This may take hours. Consider running offline/batched.")
    
    t0 = time.time()
    result = sobol_indices(n_samples=n_samples, 
                           n_ensemble_per_point=n_ensemble,
                           observable="along_track_m")
    elapsed = time.time() - t0
    print(f"\nSobol complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    
    print_sobol_results(result)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f'{output_dir}/sobol_results.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {output_dir}/sobol_results.json")
    
    return result


def run_identifiability(n_design=128, n_ensemble=8, output_dir="./phase1_results"):
    """Run duration-rho identifiability map."""
    print("\n" + "=" * 80)
    print("IDENTIFIABILITY MAP")
    print("=" * 80)
    
    from .phase1_identifiability_map import build_duration_rho_map_fast, print_identifiability_map
    
    results = build_duration_rho_map_fast(
        n_design=n_design, n_ensemble=n_ensemble, output_dir=output_dir)
    
    print_identifiability_map(results, observable='along_track_m')
    print_identifiability_map(results, observable='delta_a_m')
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Signal Existence Audit")
    parser.add_argument('--mode', 
                       choices=['pilot', 'morris', 'sobol', 'identifiability', 'full'],
                       default='pilot', help='Analysis mode')
    parser.add_argument('--n', type=int, default=None, 
                       help='Sample size (design points for pilot/identifiability, trajectories for Morris, samples for Sobol)')
    parser.add_argument('--ensemble', type=int, default=4,
                       help='Ensemble size per evaluation point')
    parser.add_argument('--output', type=str, default='./phase1_results',
                       help='Output directory')
    args = parser.parse_args()
    
    print("PHASE 1 — SIGNAL EXISTENCE AUDIT")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Output: {args.output}")
    
    if args.mode == 'pilot':
        n = args.n or 16
        run_pilot(n_design=n, n_ensemble=args.ensemble, output_dir=args.output)
    
    elif args.mode == 'morris':
        n = args.n or 20
        run_morris(n_trajectories=n, n_ensemble=args.ensemble, output_dir=args.output)
    
    elif args.mode == 'sobol':
        n = args.n or 256
        run_sobol(n_samples=n, n_ensemble=args.ensemble, output_dir=args.output)
    
    elif args.mode == 'identifiability':
        n = args.n or 128
        run_identifiability(n_design=n, n_ensemble=args.ensemble, output_dir=args.output)
    
    elif args.mode == 'full':
        n = args.n or 512
        print("Running full pipeline...")
        run_pilot(n_design=16, n_ensemble=args.ensemble, output_dir=args.output)
        run_morris(n_trajectories=20, n_ensemble=args.ensemble, output_dir=args.output)
        run_identifiability(n_design=128, n_ensemble=args.ensemble, output_dir=args.output)
        print("\nFull pipeline complete (Sobol skipped — run separately with --mode sobol)")
    
    print("\nDone.")


if __name__ == '__main__':
    main()