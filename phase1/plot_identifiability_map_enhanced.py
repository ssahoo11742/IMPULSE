"""Plot enhanced identifiability maps (works with fixed + unwrapped data).

Usage:
    python plot_identifiability_map_enhanced.py --input ./phase1_results/identifiability_map_enhanced.pkl --output ./phase1_plots_enhanced
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


def plot_heatmap(results, observable='along_track_m', output_path=None, title_suffix=""):
    ident = results['identifiability'][observable]
    durations = results['duration_days']
    rhos = results['log10_rho']

    fig, ax = plt.subplots(figsize=(14, 8))
    colors = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850']
    cmap = LinearSegmentedColormap.from_list('identifiability', colors)

    im = ax.imshow(ident, aspect='auto', origin='lower', cmap=cmap,
                   extent=[rhos[0], rhos[-1], durations[0], durations[-1]], vmin=0, vmax=2.0)

    cs = ax.contour(rhos, durations, ident, levels=[0.1, 0.3, 0.5],
                    colors=['black', 'darkblue', 'darkgreen'],
                    linewidths=[1, 1.5, 2], linestyles=['--', '-', '-'])
    ax.clabel(cs, inline=True, fontsize=9, fmt={0.1: 'floor', 0.3: 'weak', 0.5: 'strong'})

    cbar = plt.colorbar(im, ax=ax, label='Identifiability Score')
    cbar.ax.axhline(y=0.1, color='black', linestyle='--', linewidth=1)
    cbar.ax.axhline(y=0.3, color='darkblue', linestyle='-', linewidth=1.5)
    cbar.ax.axhline(y=0.5, color='darkgreen', linestyle='-', linewidth=2)

    ax.set_xlabel('log$_{10}$(Debris Density) [fragments/m$^3$]', fontsize=13)
    ax.set_ylabel('Observation Duration [days]', fontsize=13)
    ax.set_title(f'Operating Envelope: {observable}{title_suffix}', fontsize=15)
    ax.set_yscale('log')

    ax.text(0.02, 0.98, "Green = strongly identifiable\nYellow = weakly identifiable\nRed = below detection floor",
            transform=ax.transAxes, ha='left', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")


def plot_comparison_grid(results, output_path=None):
    """Plot comparison of wrapped vs unwrapped observables."""
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    colors = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850']
    cmap = LinearSegmentedColormap.from_list('identifiability', colors)
    durations = results['duration_days']
    rhos = results['log10_rho']

    observables = [
        ('along_track_m', 'Wrapped: along_track_m'),
        ('along_track_m_unwrapped', 'UNWRAPPED: along_track_m'),
        ('along_track_hp', 'Wrapped: along_track_hp'),
        ('along_track_hp_unwrapped', 'UNWRAPPED: along_track_hp'),
        ('delta_a_m', 'Reference: delta_a_m'),
        ('along_track_m_var', 'Variance: along_track_m'),
        ('along_track_m_unwrapped_var', 'Variance: along_track_m_unwrapped'),
        ('delta_ecc', 'Reference: delta_ecc'),
    ]

    for idx, (obs, title) in enumerate(observables):
        ax = axes[idx // 4, idx % 4]
        if obs not in results['identifiability']:
            ax.text(0.5, 0.5, f"{obs}\nnot computed", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue

        ident = results['identifiability'][obs]
        im = ax.imshow(ident, aspect='auto', origin='lower', cmap=cmap,
                       extent=[rhos[0], rhos[-1], durations[0], durations[-1]], vmin=0, vmax=2.0)
        ax.contour(rhos, durations, ident, levels=[0.1, 0.3, 0.5],
                   colors=['black', 'darkblue', 'darkgreen'],
                   linewidths=[0.8, 1, 1.2], linestyles=['--', '-', '-'])
        ax.set_xlabel('log$_{10}$(rho)', fontsize=10)
        ax.set_ylabel('Duration [days]', fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_yscale('log')
        plt.colorbar(im, ax=ax, label='Score', fraction=0.046, pad=0.04)

    plt.suptitle('Identifiability Comparison: Wrapped vs Unwrapped Phase', fontsize=16, y=1.02)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")


def plot_wrapped_vs_unwrapped_diff(results, output_path=None):
    """Plot the difference between wrapped and unwrapped identifiability."""
    if 'along_track_hp' not in results['identifiability'] or 'along_track_hp_unwrapped' not in results['identifiability']:
        print("Skipping diff plot: need both wrapped and unwrapped along_track_hp")
        return

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    durations = results['duration_days']
    rhos = results['log10_rho']

    ident_wrapped = results['identifiability']['along_track_hp']
    ident_unwrapped = results['identifiability']['along_track_hp_unwrapped']
    diff = ident_unwrapped - ident_wrapped

    # Wrapped
    im0 = axes[0].imshow(ident_wrapped, aspect='auto', origin='lower', cmap='RdYlGn',
                         extent=[rhos[0], rhos[-1], durations[0], durations[-1]], vmin=0, vmax=2.0)
    axes[0].set_title('Wrapped: along_track_hp', fontsize=12)
    axes[0].set_xlabel('log$_{10}$(rho)')
    axes[0].set_ylabel('Duration [days]')
    axes[0].set_yscale('log')
    plt.colorbar(im0, ax=axes[0], label='SNR')

    # Unwrapped
    im1 = axes[1].imshow(ident_unwrapped, aspect='auto', origin='lower', cmap='RdYlGn',
                         extent=[rhos[0], rhos[-1], durations[0], durations[-1]], vmin=0, vmax=2.0)
    axes[1].set_title('UNWRAPPED: along_track_hp', fontsize=12)
    axes[1].set_xlabel('log$_{10}$(rho)')
    axes[1].set_ylabel('Duration [days]')
    axes[1].set_yscale('log')
    plt.colorbar(im1, ax=axes[1], label='SNR')

    # Difference
    im2 = axes[2].imshow(diff, aspect='auto', origin='lower', cmap='coolwarm',
                         extent=[rhos[0], rhos[-1], durations[0], durations[-1]], vmin=-1, vmax=1)
    axes[2].set_title('Difference (Unwrapped - Wrapped)', fontsize=12)
    axes[2].set_xlabel('log$_{10}$(rho)')
    axes[2].set_ylabel('Duration [days]')
    axes[2].set_yscale('log')
    plt.colorbar(im2, ax=axes[2], label='ΔSNR')

    # Add contour for zero difference
    axes[2].contour(rhos, durations, diff, levels=[0], colors='black', linewidths=1)

    plt.suptitle('Aliasing Impact: Wrapped vs Unwrapped Along-Track', fontsize=14)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")


def plot_snr_comparison(results, output_path=None):
    observables = ['along_track_m', 'along_track_m_unwrapped',
                   'along_track_hp', 'along_track_hp_unwrapped',
                   'delta_a_m', 'delta_ecc']
    max_snrs, labels, colors_list = [], [], []
    color_map = {'along_track_m': '#1f77b4', 'along_track_m_unwrapped': '#aec7e8',
                 'along_track_hp': '#ff7f0e', 'along_track_hp_unwrapped': '#ffbb78',
                 'delta_a_m': '#8c564b', 'delta_ecc': '#e377c2'}

    for obs in observables:
        if obs in results['identifiability']:
            max_snrs.append(np.nanmax(results['identifiability'][obs]))
            labels.append(obs.replace('_', '\n'))
            colors_list.append(color_map.get(obs, '#333333'))

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(labels, max_snrs, color=colors_list, edgecolor='black')
    ax.axhline(y=0.1, color='black', linestyle='--', alpha=0.5, label='detection floor')
    ax.axhline(y=0.3, color='darkblue', linestyle='--', alpha=0.5, label='weak threshold')
    ax.axhline(y=0.5, color='darkgreen', linestyle='--', alpha=0.5, label='strong threshold')

    ax.set_ylabel('Max Identifiability Score (SNR)', fontsize=12)
    ax.set_title('Peak SNR Comparison: Wrapped vs Unwrapped', fontsize=14)
    ax.legend()
    ax.set_ylim(0, max(max_snrs) * 1.2 if max_snrs else 1)

    for bar, val in zip(bars, max_snrs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot Phase 1 Enhanced Identifiability Map (Unwrapped)')
    parser.add_argument('--input', type=str, default='./phase1_results/identifiability_map_enhanced.pkl')
    parser.add_argument('--output', type=str, default='./phase1_plots_enhanced')
    args = parser.parse_args()

    pkl_path = Path(args.input)
    if not pkl_path.exists():
        print(f"ERROR: File not found: {pkl_path}")
        print("Run: python -m phase1.phase1_identifiability_map --mode fast --n 128 --ensemble 8")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {pkl_path}")
    results = load_results(pkl_path)

    # Plot all key observables
    for obs in ['along_track_m', 'along_track_m_unwrapped',
                'along_track_hp', 'along_track_hp_unwrapped']:
        if obs in results['identifiability']:
            suffix = ""
            if 'unwrapped' in obs: suffix = " (UNWRAPPED)"
            elif obs == 'along_track_hp': suffix = " (Step 2: High-Pass)"
            print(f"Plotting {obs} heatmap...")
            plot_heatmap(results, observable=obs,
                        output_path=output_dir / f'identifiability_map_{obs}.png',
                        title_suffix=suffix)

    print("Plotting comparison grid...")
    plot_comparison_grid(results, output_path=output_dir / 'identifiability_comparison_grid.png')

    print("Plotting wrapped vs unwrapped diff...")
    plot_wrapped_vs_unwrapped_diff(results, output_path=output_dir / 'wrapped_vs_unwrapped_diff.png')

    print("Plotting SNR comparison...")
    plot_snr_comparison(results, output_path=output_dir / 'snr_comparison.png')

    print(f"\nAll plots saved to: {output_dir}/")

if __name__ == '__main__':
    main()