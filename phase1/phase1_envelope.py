"""Extract algorithmic operating envelope from existing identifiability map."""

import pickle
import numpy as np
import json
from pathlib import Path

def extract_envelope(pkl_path, observable='along_track_hp_unwrapped', 
                      snr_threshold=0.3, output_path=None):
    """
    Extract operating envelope from existing identifiability map.

    Uses SNR > snr_threshold as the detectability criterion.
    Coverage is approximated from non-NaN, non-zero signal strength.
    """
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)

    ident = results['identifiability'][observable]
    signal = results['signal_strength'][observable]
    durations = results['duration_days']
    rhos = results['log10_rho']

    # Approximate coverage: cells with ident > 0 had some valid runs
    approx_coverage = np.where((ident > 0) | (signal > 0), 1.0, 0.0)

    # Envelope mask: SNR >= threshold AND had valid runs
    envelope_mask = (ident >= snr_threshold) & (approx_coverage > 0)

    # Find boundary: lowest rho per duration where envelope is True
    boundary = []
    for i, dur in enumerate(durations):
        valid_j = np.where(envelope_mask[i, :])[0]
        if len(valid_j) > 0:
            j_min = valid_j[0]  # Leftmost valid rho
            boundary.append({
                'duration_days': int(dur),
                'log10_rho_min': float(rhos[j_min]),
                'rho_min': float(10 ** rhos[j_min]),
                'n_valid_rhos': int(len(valid_j)),
                'max_snr_in_row': float(np.max(ident[i, :]))
            })
        else:
            boundary.append({
                'duration_days': int(dur),
                'log10_rho_min': None,
                'rho_min': None,
                'n_valid_rhos': 0,
                'max_snr_in_row': float(np.max(ident[i, :])) if not np.all(np.isnan(ident[i, :])) else None
            })

    envelope = {
        'observable': observable,
        'snr_threshold': snr_threshold,
        'boundary': boundary,
        'durations': durations.tolist(),
        'rhos': rhos.tolist(),
        'envelope_mask': envelope_mask.tolist(),
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(envelope, f, indent=2)
        print(f"Envelope saved to: {output_path}")

    return envelope


def print_envelope_table(envelope):
    """Pretty-print the operating envelope."""
    print("\n" + "=" * 70)
    print(f"OPERATING ENVELOPE — Observable: {envelope['observable']}")
    print(f"SNR Threshold: {envelope['snr_threshold']}")
    print("=" * 70)
    print(f"{'Duration':>10s} | {'Min log₁₀ρ':>12s} | {'Min ρ (frags/m³)':>18s} | {'Max SNR':>10s}")
    print("-" * 70)

    for b in envelope['boundary']:
        dur = b['duration_days']
        rho_log = b['log10_rho_min']
        rho = b['rho_min']
        max_snr = b['max_snr_in_row']

        if rho_log is None:
            if max_snr is not None:
                print(f"{dur:>8d} d | {'N/A':>12s} | {'N/A':>18s} | {max_snr:>10.3f}")
            else:
                print(f"{dur:>8d} d | {'N/A':>12s} | {'N/A':>18s} | {'N/A':>10s}")
        else:
            print(f"{dur:>8d} d | {rho_log:>12.2f} | {rho:>18.2e} | {max_snr:>10.3f}")

    print("-" * 70)
    print("Interpretation: For given duration, debris density must be")
    print("at least 'Min ρ' for reliable detection via this observable.")


def load_envelope(path='./phase1_results/operating_envelope_along_track_hp_unwrapped_snr0.3.json'):
    """Load an operating envelope from JSON."""
    with open(path) as f:
        return json.load(f)


def is_in_envelope(params_dict, envelope):
    """
    Check if a parameter point is inside the operating envelope.
    Uses linear interpolation of the boundary.
    """
    duration = params_dict['duration_days']
    log10_rho = params_dict['log10_rho_debris']

    # Extract boundary as arrays
    durations = np.array([b['duration_days'] for b in envelope['boundary']])
    rho_mins = np.array([b['log10_rho_min'] if b['log10_rho_min'] is not None else np.inf 
                         for b in envelope['boundary']])

    # Interpolate minimum rho for this duration
    if duration < durations.min() or duration > durations.max():
        return False

    rho_min_at_dur = np.interp(duration, durations, rho_mins)

    # If interpolated boundary is inf, no detection possible at this duration
    if np.isinf(rho_min_at_dur):
        return False

    return log10_rho >= rho_min_at_dur


if __name__ == '__main__':
    import sys

    pkl_path = './phase1_results/identifiability_map_fast.pkl'
    if len(sys.argv) > 1:
        pkl_path = sys.argv[1]

    # Try multiple thresholds
    for snr_thresh in [0.3, 0.5]:
        envelope = extract_envelope(
            pkl_path, 
            observable='along_track_m_unwrapped',
            snr_threshold=snr_thresh,
            output_path=f'./phase1_results/operating_envelope_snr{snr_thresh}.json'
        )
        print_envelope_table(envelope)
        print("\n")