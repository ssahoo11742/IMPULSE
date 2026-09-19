import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

with open('results/real_tles.tle.meta.json') as f:
    meta = json.load(f)

inc = np.array([float(m['INCLINATION']) for m in meta])
alt = np.array([(float(m['PERIAPSIS']) + float(m['APOAPSIS'])) / 2 for m in meta])

PRIMARY_BLUE = '#2b5c8f'
HIGHLIGHT_ORANGE = '#d95f02'
TEXT_DARK = '#222222'

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

fig = plt.figure(figsize=(13.5, 4.2))
gs = fig.add_gridspec(1, 3, wspace=0.32)

# --- Panel A: Altitude histogram ---
axA = fig.add_subplot(gs[0, 0])
bins_alt = np.arange(700, 1021, 20)
n, b, patches = axA.hist(alt, bins=bins_alt, color=PRIMARY_BLUE, edgecolor='white', linewidth=0.5)
for p, edge in zip(patches, b[:-1]):
    if 800 <= edge < 900:
        p.set_facecolor(HIGHLIGHT_ORANGE)

axA.set_ylim(0, 480)  # Added headroom so text doesn't hit bars
axA.set_xlabel('Mean altitude (km)', fontsize=11, labelpad=6)
axA.set_ylabel('Number of spacecraft', fontsize=11, labelpad=6)
axA.set_title('(a) Altitude distribution', fontsize=12, loc='left', fontweight='bold', pad=10)

# Moved to top-right where altitude bin counts are lower
axA.text(0.96, 0.94, f'N = {len(alt):,}\nmedian = {np.median(alt):.0f} km',
         transform=axA.transAxes, va='top', ha='right', fontsize=9.5, color=TEXT_DARK,
         bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', boxstyle='round,pad=0.3'))
axA.spines[['top', 'right']].set_visible(False)
axA.grid(axis='y', linestyle='--', alpha=0.3)

# --- Panel B: Inclination histogram ---
axB = fig.add_subplot(gs[0, 1])
bins_inc = np.arange(15, 150, 5)
n, b, patches = axB.hist(inc, bins=bins_inc, color=PRIMARY_BLUE, edgecolor='white', linewidth=0.5)
for p, edge in zip(patches, b[:-1]):
    if 95 <= edge < 105:
        p.set_facecolor(HIGHLIGHT_ORANGE)

axB.axvspan(95, 105, color=HIGHLIGHT_ORANGE, alpha=0.12, zorder=0)
axB.set_ylim(0, 2100)  # Headroom above the tall peak
axB.set_xlabel('Inclination (deg)', fontsize=11, labelpad=6)
axB.set_ylabel('Number of spacecraft', fontsize=11, labelpad=6)
axB.set_title('(b) Inclination distribution', fontsize=12, loc='left', fontweight='bold', pad=10)

sso_frac = 100 * np.mean((inc >= 95) & (inc < 105))
# Moved to top-left corner away from the SSO spike
axB.text(0.04, 0.94, f'SSO (95–105°): {sso_frac:.0f}%\nmedian: {np.median(inc):.1f}°',
         transform=axB.transAxes, va='top', ha='left', fontsize=9.5, color=TEXT_DARK,
         bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', boxstyle='round,pad=0.3'))
axB.spines[['top', 'right']].set_visible(False)
axB.grid(axis='y', linestyle='--', alpha=0.3)

# --- Panel C: 2D Altitude-Inclination Coverage ---
axC = fig.add_subplot(gs[0, 2])
H, xedges, yedges = np.histogram2d(inc, alt, bins=[bins_inc, bins_alt])
Hm = np.ma.masked_where(H == 0, H)
pc = axC.pcolormesh(xedges, yedges, Hm.T, cmap='Blues',
                    norm=matplotlib.colors.LogNorm(vmin=1, vmax=H.max()))

axC.axvspan(95, 105, color=HIGHLIGHT_ORANGE, alpha=0.12, zorder=0)
cb = fig.colorbar(pc, ax=axC, pad=0.03, aspect=24)
cb.set_label('Objects per bin', fontsize=10)
cb.ax.tick_params(labelsize=8.5)

axC.set_xlabel('Inclination (deg)', fontsize=11, labelpad=6)
axC.set_ylabel('Mean altitude (km)', fontsize=11, labelpad=6)
axC.set_title('(c) Altitude-inclination coverage', fontsize=12, loc='left', fontweight='bold', pad=10)
axC.spines[['top', 'right']].set_visible(False)

# Figure Title & Output
fig.suptitle('IMPULSE V1 Fleet Composition (N = 3,328, epoch 2026-08-16)',
             fontsize=13, fontweight='bold', y=1.03)

plt.savefig('figs/fig_fleet_composition.png', dpi=300, bbox_inches='tight')
plt.show()