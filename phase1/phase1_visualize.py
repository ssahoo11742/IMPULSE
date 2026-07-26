"""Phase 1 — Visualization Tools

Generates publication-quality plots from sensitivity analysis results:
  - Morris mu*-sigma scatter plot (parameter classification)
  - Sobol S1/ST bar charts (variance decomposition)
  - Duration-rho identifiability heatmap (the operating envelope)
  - Altitude-rho slice heatmap
  - Cd-rho slice heatmap
  - Signal vs duration curves (at fixed rho)
  - Signal vs rho curves (at fixed duration)

Usage:
    python phase1_visualize.py --input ./phase1_results --output ./phase1_plots
"""

import argparse
import json
import pickle
import numpy as np
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.patches import Rectangle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not installed. Plots will not be generated.")
    print("Install with: pip install matplotlib")


def plot_morris(morris_results, output_path):
    """Morris mu*-sigma scatter plot with classification zones."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    max_mu_star = max(r['mu_star'] for r in morris_results.values())
    threshold = 0.1 * max_mu_star

    for name, r in morris_results.items():
        mu_star = r['mu_star']
        sigma = r['sigma']

        # Classification colors
        if mu_star < threshold:
            color = 'lightgray'
            marker = 'v'
            size = 60
        elif sigma > mu_star:
            color = 'coral'
            marker = 's'
            size = 120
        else:
            color = 'forestgreen'
            marker = 'o'
            size = 120

        ax.scatter(mu_star, sigma, c=color, marker=marker, s=size, edgecolors='black', linewidth=0.5)
        ax.annotate(name, (mu_star, sigma), textcoords="offset points", 
                   xytext=(8, 4), fontsize=8, ha='left')

    # Classification zones
    ax.axhline(y=threshold, color='gray', linestyle='--', alpha=0.5, label='Importance threshold')
    ax.axvline(x=threshold, color='gray', linestyle='--', alpha=0.5)

    # Zone labels
    ax.text(0.95, 0.95, 'Linear + Important', transform=ax.transAxes, 
           ha='right', va='top', fontsize=10, color='forestgreen', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(0.95, 0.05, 'Unimportant', transform=ax.transAxes, 
           ha='right', va='bottom', fontsize=10, color='gray', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(0.05, 0.95, 'Nonlinear / Interactive', transform=ax.transAxes, 
           ha='left', va='top', fontsize=10, color='coral', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('mu* (mean absolute elementary effect)', fontsize=12)
    ax.set_ylabel('sigma (standard deviation of elementary effects)', fontsize=12)
    ax.set_title('Morris Screening: Parameter Importance Classification', fontsize=14)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_sobol(sobol_result, output_path):
    """Sobol S1 and ST bar chart."""
    if not HAS_MATPLOTLIB:
        return

    S1 = sobol_result['S1']
    ST = sobol_result['ST']

    params = list(S1.keys())
    s1_vals = [S1[p] for p in params]
    st_vals = [ST[p] for p in params]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(params))
    width = 0.35

    bars1 = ax.bar(x - width/2, s1_vals, width, label='S1 (first-order)', color='steelblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, st_vals, width, label='ST (total)', color='coral', edgecolor='black', alpha=0.8)

    # Highlight rho
    rho_idx = params.index('log10_rho_debris')
    bars1[rho_idx].set_color('forestgreen')
    bars1[rho_idx].set_edgecolor('darkgreen')
    bars1[rho_idx].set_linewidth(2)
    bars2[rho_idx].set_color('lightgreen')
    bars2[rho_idx].set_edgecolor('darkgreen')
    bars2[rho_idx].set_linewidth(2)

    ax.set_xlabel('Parameter', fontsize=12)
    ax.set_ylabel('Sobol Index', fontsize=12)
    ax.set_title(f"Sobol Variance Decomposition — {sobol_result['observable']}", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(params, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, max(max(st_vals), max(s1_vals)) * 1.2)

    # Add identifiability annotation
    from phase1_sobol import compute_identifiability
    ident = compute_identifiability(sobol_result)
    ax.text(0.02, 0.98, f"rho identifiability: {ident['score']:.3f}\n({ident['status']})",
           transform=ax.transAxes, ha='left', va='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_identifiability_heatmap(results, observable='along_track_m', output_path=None):
    """Duration-rho identifiability heatmap."""
    if not HAS_MATPLOTLIB:
        return

    ident = results['identifiability'][observable]
    durations = results['duration_days']
    rhos = results['log10_rho']

    fig, ax = plt.subplots(figsize=(14, 8))

    # Custom colormap: red (no signal) -> yellow (weak) -> green (strong)
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850']
    cmap = LinearSegmentedColormap.from_list('identifiability', colors)

    im = ax.imshow(ident, aspect='auto', origin='lower', cmap=cmap,
                   extent=[rhos[0], rhos[-1], durations[0], durations[-1]],
                   vmin=0, vmax=2.0)

    # Contour lines at thresholds
    ax.contour(rhos, durations, ident, levels=[0.1, 0.3, 0.5], 
              colors=['black', 'darkblue', 'darkgreen'], 
              linewidths=[1, 1.5, 2], linestyles=['--', '-', '-'])

    # Label contours
    ax.clabel(ax.contour(rhos, durations, ident, levels=[0.1, 0.3, 0.5]),
             inline=True, fontsize=9, fmt={0.1: 'floor', 0.3: 'weak', 0.5: 'strong'})

    cbar = plt.colorbar(im, ax=ax, label='Identifiability Score')
    cbar.ax.axhline(y=0.1, color='black', linestyle='--', linewidth=1)
    cbar.ax.axhline(y=0.3, color='darkblue', linestyle='-', linewidth=1.5)
    cbar.ax.axhline(y=0.5, color='darkgreen', linestyle='-', linewidth=2)

    ax.set_xlabel('log10(Debris Density) [fragments/m^3]', fontsize=12)
    ax.set_ylabel('Observation Duration [days]', fontsize=12)
    ax.set_title(f'Operating Envelope: ρ Identifiability via {observable}', fontsize=14)
    ax.set_yscale('log')

    # Annotation
    ax.text(0.02, 0.98, 
           "Green = strongly identifiable\n"
           "Yellow = weakly identifiable\n"
           "Red = below detection floor",
           transform=ax.transAxes, ha='left', va='top', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    else:
        plt.show()


def plot_signal_vs_duration(results, observable='along_track_m', output_path=None):
    """Signal strength vs duration at different rho values."""
    if not HAS_MATPLOTLIB:
        return

    signal = results['signal_strength'][observable]
    noise = results['nuisance_strength'][observable] + results['seed_noise'][observable]
    durations = results['duration_days']
    rhos = results['log10_rho']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Signal magnitude
    for j in [0, len(rhos)//2, len(rhos)-1]:
        ax1.plot(durations, signal[:, j], marker='o', label=f"log10 ρ = {rhos[j]:.1f}")
    ax1.set_xlabel('Duration [days]', fontsize=12)
    ax1.set_ylabel(f'Signal: |{observable}|', fontsize=12)
    ax1.set_title('Signal Magnitude vs Duration', fontsize=14)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: SNR = signal / noise
    snr = signal / (noise + 1e-12)
    for j in [0, len(rhos)//2, len(rhos)-1]:
        ax2.plot(durations, snr[:, j], marker='o', label=f"log10 ρ = {rhos[j]:.1f}")
    ax2.axhline(y=0.1, color='red', linestyle='--', label='detection floor')
    ax2.axhline(y=0.3, color='orange', linestyle='--', label='weak threshold')
    ax2.axhline(y=0.5, color='green', linestyle='--', label='strong threshold')
    ax2.set_xlabel('Duration [days]', fontsize=12)
    ax2.set_ylabel('Identifiability (SNR)', fontsize=12)
    ax2.set_title('Identifiability vs Duration', fontsize=14)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    else:
        plt.show()


def plot_all(input_dir='./phase1_results', output_dir='./phase1_plots'):
    """Generate all plots from saved results."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Morris plot
    morris_path = Path(input_dir) / 'morris_results.json'
    if morris_path.exists():
        with open(morris_path) as f:
            morris = json.load(f)
        plot_morris(morris, f'{output_dir}/morris_screening.png')

    # Sobol plot
    sobol_path = Path(input_dir) / 'sobol_results.json'
    if sobol_path.exists():
        with open(sobol_path) as f:
            sobol = json.load(f)
        plot_sobol(sobol, f'{output_dir}/sobol_decomposition.png')

    # Identifiability heatmap
    ident_path = Path(input_dir) / 'identifiability_map_fast.pkl'
    if ident_path.exists():
        with open(ident_path, 'rb') as f:
            ident = pickle.load(f)
        plot_identifiability_heatmap(ident, 'along_track_m', 
                                     f'{output_dir}/identifiability_map_along_track.png')
        plot_identifiability_heatmap(ident, 'delta_a_m',
                                     f'{output_dir}/identifiability_map_delta_a.png')
        plot_signal_vs_duration(ident, 'along_track_m',
                               f'{output_dir}/signal_vs_duration.png')

    print(f"\nAll plots saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Phase 1 Visualization')
    parser.add_argument('--input', type=str, default='./phase1_results',
                       help='Input directory with results')
    parser.add_argument('--output', type=str, default='./phase1_plots',
                       help='Output directory for plots')
    args = parser.parse_args()

    if not HAS_MATPLOTLIB:
        print("ERROR: matplotlib required for visualization.")
        print("Install: pip install matplotlib")
        return

    plot_all(args.input, args.output)


if __name__ == '__main__':
    main()