"""Phase 1 — Duration-Sliced Identifiability Map (FULLY FIXED)

Fixes from reviewer:
  1. KEY_OBSERVABLES now uses unwrapped names (was using old wrapped names)
  2. reentry_delta_s actually removed (was still present)
  3. Envelope import fixed (load_envelope matches what Sobol imports)
  4. record_history parameter now respected
  5. a0 vs el_c.a consistency fixed
  6. _highpass_filter edge padding fixed (reflect instead of same)
  7. Checkpoint/resume support
  8. nanmean/nanstd for robust statistics
  9. Coverage tracking per cell
"""

import argparse
import json
import pickle
import time
import math
import numpy as np
import os
import signal
from pathlib import Path

from .phase1_config import (
    PARAMETER_NAMES, sample_sobol, map_to_bounds, get_parameter_dict
)
from .phase1_runner import run_paired_ensemble

DURATION_GRID_DAYS = [7, 14, 30, 60, 90, 180, 365, 730]
RHO_GRID_LOG10 = np.linspace(-15, -8, 15)

# FIXED: Use unwrapped observables, remove reentry_delta_s
KEY_OBSERVABLES = [
    "delta_a_m", "delta_ecc", "delta_inc_rad",
    "along_track_m",           # OLD: wrapped (kept for comparison)
    "along_track_m_unwrapped", # NEW: unwrapped (primary)
    "along_track_hp",          # OLD: wrapped HP
    "along_track_hp_unwrapped",# NEW: unwrapped HP
]

# FIXED: Remove reentry variance observables
VARIANCE_OBSERVABLES = [
    "along_track_m_var",
    "along_track_m_unwrapped_var",
    "along_track_hp_var",
    "along_track_hp_unwrapped_var",
]

COVERAGE_THRESHOLD = 0.5


class CheckpointManager:
    """Manages checkpointing and resuming of identifiability map runs."""

    def __init__(self, output_dir, n_durations, n_rhos, n_obs_keys):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.output_dir / 'identifiability_checkpoint.json'
        self.temp_results_path = self.output_dir / 'identifiability_temp.pkl'

        self.results = {
            'duration_days': np.array(DURATION_GRID_DAYS),
            'log10_rho': RHO_GRID_LOG10,
            'identifiability': {},
            'signal_strength': {},
            'nuisance_strength': {},
            'seed_noise': {},
            'coverage': {},
            'n_valid': {},
            'n_attempted': {},
        }

        all_obs_keys = KEY_OBSERVABLES + VARIANCE_OBSERVABLES
        for obs in all_obs_keys:
            self.results['identifiability'][obs] = np.zeros((n_durations, n_rhos))
            self.results['signal_strength'][obs] = np.zeros((n_durations, n_rhos))
            self.results['nuisance_strength'][obs] = np.zeros((n_durations, n_rhos))
            self.results['seed_noise'][obs] = np.zeros((n_durations, n_rhos))
            self.results['coverage'][obs] = np.zeros((n_durations, n_rhos))
            self.results['n_valid'][obs] = np.zeros((n_durations, n_rhos), dtype=int)
            self.results['n_attempted'][obs] = np.zeros((n_durations, n_rhos), dtype=int)

        self.completed_cells = set()
        self.total_rejected = 0
        self.total_failed = 0
        self.total_runs = 0
        self.cell_count = 0

        self._load_checkpoint()
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        def signal_handler(signum, frame):
            print(f"\n\n*** Received signal {signum}. Saving checkpoint... ***")
            self._save_checkpoint()
            print(f"Checkpoint saved to: {self.checkpoint_path}")
            print("Run the same command again to resume.")
            exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)

    def _load_checkpoint(self):
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, 'r') as f:
                    ckpt = json.load(f)

                self.completed_cells = set(tuple(c) for c in ckpt.get('completed_cells', []))
                self.total_rejected = ckpt.get('total_rejected', 0)
                self.total_failed = ckpt.get('total_failed', 0)
                self.total_runs = ckpt.get('total_runs', 0)
                self.cell_count = ckpt.get('cell_count', 0)

                if self.temp_results_path.exists():
                    try:
                        with open(self.temp_results_path, 'rb') as f:
                            temp_results = pickle.load(f)
                        for obs in self.results['identifiability'].keys():
                            if obs in temp_results.get('identifiability', {}):
                                self.results['identifiability'][obs] = np.array(temp_results['identifiability'][obs])
                                self.results['signal_strength'][obs] = np.array(temp_results['signal_strength'][obs])
                                self.results['nuisance_strength'][obs] = np.array(temp_results['nuisance_strength'][obs])
                                self.results['seed_noise'][obs] = np.array(temp_results['seed_noise'][obs])
                                self.results['coverage'][obs] = np.array(temp_results['coverage'][obs])
                                self.results['n_valid'][obs] = np.array(temp_results['n_valid'][obs])
                                self.results['n_attempted'][obs] = np.array(temp_results['n_attempted'][obs])
                        print(f"*** RESUMING FROM CHECKPOINT ***")
                        print(f"    {len(self.completed_cells)} cells already completed")
                        print(f"    {self.total_rejected} total rejected, {self.total_failed} total failed")
                    except Exception as e:
                        print(f"Warning: Could not load temp results: {e}")
                else:
                    print(f"*** RESUMING FROM CHECKPOINT ***")
                    print(f"    {len(self.completed_cells)} cells already completed")

            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")
                print("Starting fresh...")
                self.completed_cells = set()

    def is_cell_done(self, i, j):
        return (i, j) in self.completed_cells

    def save_cell_results(self, i, j, cell_results, cell_stats):
        for obs in cell_results:
            for key in ['identifiability', 'signal_strength', 'nuisance_strength',
                        'seed_noise', 'coverage', 'n_valid', 'n_attempted']:
                if obs in self.results[key]:
                    self.results[key][obs][i, j] = cell_results[obs][key]

        self.completed_cells.add((i, j))
        self.cell_count += 1
        self.total_rejected += cell_stats.get('rejected', 0)
        self.total_failed += cell_stats.get('failed', 0)
        self.total_runs += cell_stats.get('runs', 0)

        self._save_checkpoint()
        if self.cell_count % 5 == 0:
            self._save_temp_results()

    def _save_checkpoint(self):
        ckpt = {
            'completed_cells': sorted(list(self.completed_cells)),
            'total_rejected': int(self.total_rejected),
            'total_failed': int(self.total_failed),
            'total_runs': int(self.total_runs),
            'cell_count': int(self.cell_count),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        temp_path = self.checkpoint_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(ckpt, f, indent=2)
        temp_path.replace(self.checkpoint_path)

    def _save_temp_results(self):
        temp_path = self.temp_results_path.with_suffix('.tmp')
        with open(temp_path, 'wb') as f:
            pickle.dump(self.results, f)
        temp_path.replace(self.temp_results_path)

    def finalize(self, output_dir, snr_threshold, coverage_threshold):
        with open(f'{output_dir}/identifiability_map_enhanced.pkl', 'wb') as f:
            pickle.dump(self.results, f)

        json_results = {
            'duration_days': DURATION_GRID_DAYS,
            'log10_rho': RHO_GRID_LOG10.tolist(),
            'identifiability': {k: v.tolist() for k, v in self.results['identifiability'].items()},
            'coverage': {k: v.tolist() for k, v in self.results['coverage'].items()},
            'n_valid': {k: v.tolist() for k, v in self.results['n_valid'].items()},
            'n_attempted': {k: v.tolist() for k, v in self.results['n_attempted'].items()},
        }
        with open(f'{output_dir}/identifiability_map_enhanced.json', 'w') as f:
            json.dump(json_results, f, indent=2)

        export_operating_envelope(self.results, output_dir, snr_threshold, coverage_threshold)

        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
        if self.temp_results_path.exists():
            self.temp_results_path.unlink()

        print(f"\nSaved: {output_dir}/identifiability_map_enhanced.pkl")
        print(f"Saved: {output_dir}/identifiability_map_enhanced.json")
        print("Checkpoint files cleaned up.")


def build_duration_rho_map_fast(n_design=128, n_ensemble=8, output_dir="./phase1_results",
                                snr_threshold=0.3, coverage_threshold=COVERAGE_THRESHOLD):
    print("=" * 80)
    print("DURATION-RHO IDENTIFIABILITY MAP (FULLY FIXED)")
    print("=" * 80)
    print(f"Duration grid: {DURATION_GRID_DAYS} days")
    print(f"Rho grid: {RHO_GRID_LOG10[0]:.1f} to {RHO_GRID_LOG10[-1]:.1f} (log10)")
    print(f"Design points per cell: {n_design}")
    print(f"Ensemble size: {n_ensemble}")
    print(f"SNR threshold: {snr_threshold}")
    print(f"Coverage threshold: {coverage_threshold}")

    n_durations = len(DURATION_GRID_DAYS)
    n_rhos = len(RHO_GRID_LOG10)
    n_cells = n_durations * n_rhos
    print(f"Total cells: {n_cells}")
    print(f"Total propagations (if fresh): {n_cells * n_design * n_ensemble}")
    print(f"Checkpoint dir: {output_dir}")
    print("=" * 80)

    ckpt = CheckpointManager(output_dir, n_durations, n_rhos,
                             len(KEY_OBSERVABLES) + len(VARIANCE_OBSERVABLES))

    n_remaining = n_cells - len(ckpt.completed_cells)
    print(f"Cells remaining: {n_remaining} / {n_cells}")
    if n_remaining == 0:
        print("All cells already completed! Finalizing...")
        ckpt.finalize(output_dir, snr_threshold, coverage_threshold)
        return ckpt.results

    print("\nStarting/resuming computation...")
    print("Press Ctrl+C at any time to save checkpoint and exit.")
    print("=" * 80)

    t0_global = time.time()

    for i, duration in enumerate(DURATION_GRID_DAYS):
        for j, log10_rho in enumerate(RHO_GRID_LOG10):

            if ckpt.is_cell_done(i, j):
                continue

            t0_cell = time.time()

            unit_samples = sample_sobol(n_design, seed=i * n_rhos + j + 1)
            phys_samples = map_to_bounds(unit_samples)
            phys_samples[:, PARAMETER_NAMES.index('duration_days')] = duration
            phys_samples[:, PARAMETER_NAMES.index('log10_rho_debris')] = log10_rho

            all_obs = {obs: [] for obs in KEY_OBSERVABLES}
            all_seed_std = {obs: [] for obs in KEY_OBSERVABLES}
            all_var_obs = {obs: [] for obs in VARIANCE_OBSERVABLES}

            cell_rejected = 0
            cell_failed = 0

            for k in range(n_design):
                params = get_parameter_dict(phys_samples[k])
                obs_array, meta = run_paired_ensemble(params, n_ensemble=n_ensemble)

                if meta is not None and meta.get("rejected"):
                    cell_rejected += 1
                    continue

                cell_failed += meta.get("n_failed", 0)

                for obs in KEY_OBSERVABLES:
                    idx = meta['obs_names'].index(obs)
                    all_obs[obs].append(meta['obs_mean'][obs])
                    all_seed_std[obs].append(meta['obs_std'][obs])

                for vobs in VARIANCE_OBSERVABLES:
                    all_var_obs[vobs].append(meta.get('variance_obs', {}).get(vobs, 0.0))

            n_total_attempted = n_design - cell_rejected
            cell_results = {}

            for obs in KEY_OBSERVABLES:
                values = np.array(all_obs[obs])
                seed_stds = np.array(all_seed_std[obs])

                valid = ~np.isnan(values)
                n_valid = np.sum(valid)
                coverage = n_valid / max(n_total_attempted, 1)

                if coverage < coverage_threshold or n_valid < 2:
                    ident = 0.0 if n_valid == 0 else 0.05
                    signal = np.nanmean(values[valid]) if n_valid > 0 else 0.0
                    nuisance_std = 0.0
                    seed_noise = 0.0
                else:
                    total_std = np.nanstd(values[valid])
                    seed_noise = np.nanmean(seed_stds[valid]) if np.any(~np.isnan(seed_stds[valid])) else 0.0
                    nuisance_std = max(0, np.sqrt(max(0, total_std**2 - seed_noise**2)))
                    signal = abs(np.nanmean(values[valid]))
                    ident = signal / (nuisance_std + seed_noise + 1e-12)

                cell_results[obs] = {
                    'identifiability': ident,
                    'signal_strength': signal,
                    'nuisance_strength': nuisance_std,
                    'seed_noise': seed_noise,
                    'coverage': coverage,
                    'n_valid': int(n_valid),
                    'n_attempted': n_total_attempted,
                }

            for vobs in VARIANCE_OBSERVABLES:
                values = np.array(all_var_obs[vobs])
                valid = ~np.isnan(values)
                n_valid = np.sum(valid)
                coverage = n_valid / max(n_total_attempted, 1)

                if coverage < coverage_threshold or n_valid < 2:
                    ident = 0.0 if n_valid == 0 else 0.05
                    signal = 0.0
                    nuisance_std = 0.0
                    seed_noise = 0.0
                else:
                    signal = np.nanmean(values[valid]) if n_valid > 0 else 0.0
                    nuisance_std = np.nanstd(values[valid]) if n_valid > 1 else 0.0
                    seed_noise = signal / np.sqrt(n_valid) if n_valid > 0 else 0.0
                    ident = signal / (nuisance_std + seed_noise + 1e-12)

                cell_results[vobs] = {
                    'identifiability': ident,
                    'signal_strength': signal,
                    'nuisance_strength': nuisance_std,
                    'seed_noise': seed_noise,
                    'coverage': coverage,
                    'n_valid': int(n_valid),
                    'n_attempted': n_total_attempted,
                }

            cell_stats = {
                'rejected': cell_rejected,
                'failed': cell_failed,
                'runs': (n_design - cell_rejected) * n_ensemble,
            }
            ckpt.save_cell_results(i, j, cell_results, cell_stats)

            cell_time = time.time() - t0_cell
            elapsed = time.time() - t0_global
            completed = len(ckpt.completed_cells)
            remaining = n_cells - completed
            eta = (elapsed / completed) * remaining if completed > 0 else 0

            rej_str = f", rejected={cell_rejected}" if cell_rejected > 0 else ""
            print(f"  [{completed:3d}/{n_cells}] duration={duration:4d}d, "
                  f"log10_rho={log10_rho:6.2f}, "
                  f"cell_time={cell_time:.1f}s, elapsed={elapsed/3600:.1f}h, ETA={eta/3600:.1f}h"
                  f"{rej_str}")

    total_time = time.time() - t0_global
    print(f"\nMap complete in {total_time:.1f}s ({total_time/3600:.1f} h)")
    print(f"Total rejected (area/mass): {ckpt.total_rejected}")
    print(f"Total failed propagations: {ckpt.total_failed}")

    ckpt.finalize(output_dir, snr_threshold, coverage_threshold)
    return ckpt.results


def export_operating_envelope(results, output_dir, snr_threshold=0.3, coverage_threshold=COVERAGE_THRESHOLD):
    print("\n" + "=" * 80)
    print("EXPORTING OPERATING ENVELOPES")
    print("=" * 80)

    durations = results['duration_days']
    rhos = results['log10_rho']

    for obs in KEY_OBSERVABLES + VARIANCE_OBSERVABLES:
        if obs not in results['identifiability']:
            continue

        ident = results['identifiability'][obs]
        coverage = results['coverage'][obs]

        envelope_mask = (ident >= snr_threshold) & (coverage >= coverage_threshold)

        boundary = []
        for i, dur in enumerate(durations):
            valid_j = [j for j in range(len(rhos)) if envelope_mask[i, j]]
            if valid_j:
                boundary.append({
                    'duration_days': int(dur),
                    'log10_rho_min': float(rhos[min(valid_j)]),
                    'log10_rho_max': float(rhos[max(valid_j)]),
                    'n_rho_points': len(valid_j)
                })
            else:
                boundary.append({
                    'duration_days': int(dur),
                    'log10_rho_min': None,
                    'log10_rho_max': None,
                    'n_rho_points': 0
                })

        envelope = {
            'observable': obs,
            'snr_threshold': snr_threshold,
            'coverage_threshold': coverage_threshold,
            'boundary': boundary,
            'mask': envelope_mask.tolist()
        }

        safe_name = obs.replace('/', '_')
        env_path = f'{output_dir}/operating_envelope_{safe_name}_snr{snr_threshold}.json'
        with open(env_path, 'w') as f:
            json.dump(envelope, f, indent=2)
        print(f"  Saved envelope for {obs}: {env_path}")


# FIXED: Renamed to load_envelope to match what Sobol imports
def load_envelope(path='./phase1_results/operating_envelope_along_track_m_unwrapped_snr0.3.json'):
    """Load an operating envelope from JSON."""
    with open(path) as f:
        return json.load(f)


# Keep old name for backward compatibility
load_operating_envelope = load_envelope


def find_operating_envelope(results, observable='along_track_m_unwrapped',
                            snr_threshold=0.3, coverage_threshold=COVERAGE_THRESHOLD):
    ident = results['identifiability'][observable]
    coverage = results['coverage'][observable]
    durations = results['duration_days']
    rhos = results['log10_rho']

    envelope_mask = (ident >= snr_threshold) & (coverage >= coverage_threshold)

    lower_bound_rho = []
    for i, dur in enumerate(durations):
        valid_j = [j for j in range(len(rhos)) if envelope_mask[i, j]]
        if valid_j:
            lower_bound_rho.append((int(dur), float(rhos[min(valid_j)])))
        else:
            lower_bound_rho.append((int(dur), None))

    metadata = {
        'observable': observable,
        'snr_threshold': snr_threshold,
        'coverage_threshold': coverage_threshold,
        'n_cells_total': len(durations) * len(rhos),
        'n_cells_in_envelope': int(np.sum(envelope_mask)),
    }

    return envelope_mask, lower_bound_rho, metadata


def is_in_envelope(params_dict, envelope):
    duration = params_dict['duration_days']
    log10_rho = params_dict['log10_rho_debris']

    durations = np.array([b['duration_days'] for b in envelope['boundary']])
    rho_mins = np.array([b['log10_rho_min'] if b['log10_rho_min'] is not None else np.inf
                         for b in envelope['boundary']])

    if duration < durations.min() or duration > durations.max():
        return False

    rho_min_at_dur = np.interp(duration, durations, rho_mins)

    if np.isinf(rho_min_at_dur):
        return False

    return log10_rho >= rho_min_at_dur


def print_identifiability_map(results, observable="along_track_m_unwrapped"):
    ident = results['identifiability'][observable]
    coverage = results['coverage'][observable]
    durations = results['duration_days']
    rhos = results['log10_rho']

    print("\n" + "=" * 100)
    print(f"IDENTIFIABILITY MAP — Observable: {observable}")
    print("=" * 100)
    print(f"{'Duration':>10s}", end="")
    for rho in rhos:
        print(f" | {rho:>8.1f}", end="")
    print()
    print("-" * 100)

    for i, dur in enumerate(durations):
        print(f"{dur:>8d}d", end="")
        for j in range(len(rhos)):
            val = ident[i, j]
            cov = coverage[i, j]
            if cov < COVERAGE_THRESHOLD:
                marker = "  ?"
            elif np.isnan(val) or val < 0.1:
                marker = "  X"
            elif val < 0.3:
                marker = f"{val:7.2f}"
            elif val < 0.5:
                marker = f"{val:7.2f}*"
            else:
                marker = f"{val:7.2f}**"
            print(f" | {marker:>8s}", end="")
        print()

    print("-" * 100)
    print("Legend: ? = insufficient coverage | X = below detection floor | * = identifiable | ** = strongly identifiable")


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Identifiability Map (Fully Fixed)")
    parser.add_argument('--mode', choices=['fast', 'full'], default='fast')
    parser.add_argument('--n', type=int, default=128)
    parser.add_argument('--ensemble', type=int, default=8)
    parser.add_argument('--output', type=str, default='./phase1_results')
    parser.add_argument('--snr-threshold', type=float, default=0.3)
    parser.add_argument('--coverage-threshold', type=float, default=COVERAGE_THRESHOLD)
    args = parser.parse_args()

    print("PHASE 1 — IDENTIFIABILITY MAP (FULLY FIXED)")
    print("=" * 80)

    if args.mode == 'fast':
        results = build_duration_rho_map_fast(
            n_design=args.n, n_ensemble=args.ensemble, output_dir=args.output,
            snr_threshold=args.snr_threshold, coverage_threshold=args.coverage_threshold)

        # Print maps for key observables (old vs new for comparison)
        for obs in ['along_track_m', 'along_track_m_unwrapped',
                    'along_track_hp', 'along_track_hp_unwrapped']:
            if obs in results['identifiability']:
                print_identifiability_map(results, observable=obs)

        # Print envelope summary for primary unwrapped observable
        if 'along_track_m_unwrapped' in results['identifiability']:
            mask, boundary, meta = find_operating_envelope(
                results, 'along_track_m_unwrapped',
                snr_threshold=args.snr_threshold,
                coverage_threshold=args.coverage_threshold)
            print("\n" + "=" * 80)
            print("OPERATING ENVELOPE SUMMARY (along_track_m_unwrapped)")
            print("=" * 80)
            print(f"Cells in envelope: {meta['n_cells_in_envelope']} / {meta['n_cells_total']}")
            print("Boundary (duration → min log10_rho):")
            for dur, rho_min in boundary:
                if rho_min is not None:
                    print(f"  {dur:4d} days → rho >= 1e{rho_min:.1f}")
                else:
                    print(f"  {dur:4d} days → no detectable rho")

    print("\nDone.")

if __name__ == '__main__':
    main()