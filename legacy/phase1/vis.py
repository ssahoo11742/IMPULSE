"""Plot identifiability map from Phase 1 results.

Usage:
    python plot_identifiability_map.py --input ./phase1_results/identifiability_map_fast.pkl --output ./phase1_plots
"""

import argparse
import pickle
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def load_results(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def plot_heatmap(results, observable='along_track_m', output_path=None):
    ident = results['identifiability'][observable]
    durations = results['duration_days']
    rhos = results['log10_rho']

    fig, ax = plt.subplots(figsize=(14, 8))

    # Custom colormap: red -> yellow -> green
    colors = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850']
    cmap = LinearSegmentedColormap.from_list('identifiability', colors)

    # Use imshow with extent to map array indices to physical coordinates
    im = ax.imshow(
        ident,
        aspect='auto',
        origin='lower',
        cmap=cmap,
        extent=[rhos[0], rhos[-1], durations[0], durations[-1]],
        vmin=0,
        vmax=2.0,
    )

    # Contour lines at thresholds
    cs = ax.contour(
        rhos,
        durations,
        ident,
        levels=[0.1, 0.3, 0.5],
        colors=['black', 'darkblue', 'darkgreen'],
        linewidths=[1, 1.5, 2],
        linestyles=['--', '-', '-'],
    )
    ax.clabel(cs, inline=True, fontsize=9, fmt={0.1: 'floor', 0.3: 'weak', 0.5: 'strong'})

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Identifiability Score')
    cbar.ax.axhline(y=0.1, color='black', linestyle='--', linewidth=1)
    cbar.ax.axhline(y=0.3, color='darkblue', linestyle='-', linewidth=1.5)
    cbar.ax.axhline(y=0.5, color='darkgreen', linestyle='-', linewidth=2)

    # Labels
    ax.set_xlabel('log$_{10}$(Debris Density) [fragments/m$^3$]', fontsize=13)
    ax.set_ylabel('Observation Duration [days]', fontsize=13)
    ax.set_title(f'Operating Envelope: rho Identifiability via {observable}', fontsize=15)
    ax.set_yscale('log')

    # Annotation box
    ax.text(
        0.02, 0.98,
        "Green = strongly identifiable\n"
        "Yellow = weakly identifiable\n"
        "Red = below detection floor",
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    else:
        plt.show()


def plot_all_observables_comparison(results, output_path=None):
    """Side-by-side heatmaps for all four observables."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()

    colors = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850']
    cmap = LinearSegmentedColormap.from_list('identifiability', colors)

    durations = results['duration_days']
    rhos = results['log10_rho']

    for idx, obs in enumerate(results['identifiability'].keys()):
        ax = axes[idx]
        ident = results['identifiability'][obs]

        im = ax.imshow(
            ident,
            aspect='auto',
            origin='lower',
            cmap=cmap,
            extent=[rhos[0], rhos[-1], durations[0], durations[-1]],
            vmin=0,
            vmax=2.0,
        )

        ax.contour(
            rhos, durations, ident,
            levels=[0.1, 0.3, 0.5],
            colors=['black', 'darkblue', 'darkgreen'],
            linewidths=[0.8, 1, 1.2],
            linestyles=['--', '-', '-'],
        )

        ax.set_xlabel('log$_{10}$(rho)', fontsize=11)
        ax.set_ylabel('Duration [days]', fontsize=11)
        ax.set_title(f'{obs}', fontsize=13)
        ax.set_yscale('log')

        plt.colorbar(im, ax=ax, label='Score', fraction=0.046, pad=0.04)

    plt.suptitle('Identifiability Comparison Across All Observables', fontsize=16, y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    else:
        plt.show()


def plot_signal_vs_duration(results, observable='along_track_m', output_path=None):
    """Plot signal and identifiability vs duration at fixed rho values."""
    signal = results['signal_strength'][observable]
    noise = results['nuisance_strength'][observable] + results['seed_noise'][observable]
    durations = results['duration_days']
    rhos = results['log10_rho']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: raw signal magnitude
    for j in [0, len(rhos)//2, len(rhos)-3, len(rhos)-1]:
        ax1.plot(durations, signal[:, j], marker='o', markersize=5, linewidth=2,
                label=f"log$_{10}$ rho = {rhos[j]:.1f}")
    ax1.set_xlabel('Duration [days]', fontsize=12)
    ax1.set_ylabel(f'Signal: |{observable}| [m]', fontsize=12)
    ax1.set_title('Signal Magnitude vs Duration', fontsize=14)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: identifiability (SNR)
    snr = signal / (noise + 1e-12)
    for j in [0, len(rhos)//2, len(rhos)-3, len(rhos)-1]:
        ax2.plot(durations, snr[:, j], marker='o', markersize=5, linewidth=2,
                label=f"log$_{10}$ rho = {rhos[j]:.1f}")

    ax2.axhline(y=0.1, color='red', linestyle='--', linewidth=1.5, label='detection floor')
    ax2.axhline(y=0.3, color='orange', linestyle='--', linewidth=1.5, label='weak threshold')
    ax2.axhline(y=0.5, color='green', linestyle='--', linewidth=1.5, label='strong threshold')

    ax2.set_xlabel('Duration [days]', fontsize=12)
    ax2.set_ylabel('Identifiability (SNR)', fontsize=12)
    ax2.set_title('Identifiability vs Duration', fontsize=14)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot Phase 1 Identifiability Map')
    parser.add_argument('--input', type=str, default='./phase1_results/identifiability_map_fast.pkl',
                       help='Path to identifiability_map_fast.pkl')
    parser.add_argument('--output', type=str, default='./phase1_plots',
                       help='Output directory for plots')
    args = parser.parse_args()

    pkl_path = Path(args.input)
    if not pkl_path.exists():
        print(f"ERROR: File not found: {pkl_path}")
        print(f"Run the identifiability map first:")
        print(f"  python -m phase1.phase1_main --mode identifiability --n 128 --ensemble 8")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {pkl_path}")
    results = load_results(pkl_path)

    print(f"Durations: {results['duration_days']}")
    print(f"Rho range: {results['log10_rho'][0]:.1f} to {results['log10_rho'][-1]:.1f}")
    print(f"Observables: {list(results['identifiability'].keys())}")
    print()

    # Main heatmap: along_track_m
    print("Plotting along_track_m heatmap...")
    plot_heatmap(results, observable='along_track_m',
                output_path=output_dir / 'identifiability_map_along_track.png')

    # Secondary heatmap: delta_a_m
    print("Plotting delta_a_m heatmap...")
    plot_heatmap(results, observable='delta_a_m',
                output_path=output_dir / 'identifiability_map_delta_a.png')

    # All four observables side-by-side
    print("Plotting all observables comparison...")
    plot_all_observables_comparison(results,
                                   output_path=output_dir / 'identifiability_all_observables.png')

    # Signal vs duration curves
    print("Plotting signal vs duration curves...")
    plot_signal_vs_duration(results, observable='along_track_m',
                           output_path=output_dir / 'signal_vs_duration.png')

    print(f"All plots saved to: {output_dir}/")
    print(f"  - identifiability_map_along_track.png")
    print(f"  - identifiability_map_delta_a.png")
    print(f"  - identifiability_all_observables.png")
    print(f"  - signal_vs_duration.png")


if __name__ == '__main__':
    main()