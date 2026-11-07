"""
Patched figure script for the final paper (figs 1-7). MERGED VERSION.

Base: the user's latest fig1-3-supp script (improved extract_detected with
lowercase/total_detected handling, plain-text fig1 threshold box), plus the
real-TLE additions:

- Loads the real-TLE strike-injection curve
  (results/real_tle_efficiency_eta417.json; schema:
   {"efficiency": {"<dv>": {"epsilon": ..., "n_scored": ...}}, "dv50": ...,
    "dv90": ..., "negligible_dv_floor": {...}}).
- FIG 5: real-TLE injection efficiency curve alone.
- FIG 6: synthetic impulse sweep alone (same styling as fig5, so the
  synthetic and real-TLE curves can be shown as a matched pair).
- FIG 7: synthetic vs real-TLE overlaid, both dv50s marked, summary box.
- fig3 / fig4 auto-upgrade when the two expected-detections runs land.
- If the real-TLE JSON is absent, figs 5/7 are skipped with a warning
  (fig6 synthetic-only still renders).
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.interpolate import PchipInterpolator

os.makedirs('figs', exist_ok=True)

# ============================================================
# Current paper numbers (no legacy values)
# ============================================================
ETA = 417.94                 # design threshold, p99.9 of pooled synthetic null
FALSE_ALARM = 0.001          # 0.1% per window (design false-alarm rate)
TOTAL_WINDOWS = 15760         # valid post-exclusion fleet windows
NOISE_EXPECTED = FALSE_ALARM * TOTAL_WINDOWS   # 15.76
OBSERVED = 6                 # fleet triggers at eta (0.038%)
DV50_FALLBACK = 1.25152      # verified print from the eval loader

SWEEP_FILES = [
    'results/gate4_efficiency_eta417.json',
    'results/gate4_efficiency_flat.json',
]
REAL_TLE_FILES = [
    'results/real_tle_efficiency_eta417.json',
    'results/real_tle_efficiency.json',
]
EXPECTED_FILES = {
    'results/fleet_expected_detected_eta417.json': '1 mm-1 cm',
    'results/fleet_expected_detected_1to10cm.json': '1-10 cm',
}


# ============================================================
# Loaders
# ============================================================
def load_efficiency_curve():
    """Return (path, dv, eps, file_dv50) for the operative sweep.

    Handles both schemas: top-level {'efficiency': {...}, 'dv50': ...} and a
    flat {dv: value} map; values may be floats or dicts with 'epsilon'.
    """
    for path in SWEEP_FILES:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        eff = None
        if isinstance(data, dict) and isinstance(data.get('efficiency'), dict):
            eff = data['efficiency']
        elif isinstance(data, dict):
            eff = data  # maybe the whole file is a flat dv -> value map
        pts = []
        for k, v in eff.items():
            try:
                dv = float(k)
            except (TypeError, ValueError):
                continue
            if isinstance(v, dict):
                eps = v.get('epsilon', v.get('eps'))
            else:
                eps = v
            if eps is None:
                continue
            pts.append((dv, float(eps)))
        if pts:
            pts.sort()
            file_dv50 = None
            if isinstance(data, dict) and data.get('dv50') is not None:
                try:
                    file_dv50 = float(data['dv50'])
                except (TypeError, ValueError):
                    pass
            return path, np.array([p[0] for p in pts]), np.array([p[1] for p in pts]), file_dv50
    raise FileNotFoundError('No impulse-sweep file found in: ' + ', '.join(SWEEP_FILES))


def load_real_tle_curve():
    """Return (path, dv, eps, n_scored, floor, file_dv50, file_dv90) for the
    real-TLE strike-injection run, or None if no usable file exists.

    Schema written by real_tle_strike_injection:
      efficiency: {"<dv>": {"epsilon": float|None, "n_scored": int, ...}}
      negligible_dv_floor: {"dv": ..., "epsilon": ..., "n_scored": ...}
    """
    for path in REAL_TLE_FILES:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        if not (isinstance(data, dict) and isinstance(data.get('efficiency'), dict)):
            print(f'  [real-TLE] {path} has no "efficiency" dict; skipped')
            continue
        pts, ns = [], []
        for k, v in data['efficiency'].items():
            try:
                dv = float(k)
            except (TypeError, ValueError):
                continue
            if not isinstance(v, dict) or v.get('epsilon') is None:
                continue
            pts.append((dv, float(v['epsilon'])))
            ns.append(int(v.get('n_scored', 0)))
        pts.sort()
        if not pts:
            print(f'  [real-TLE] {path} contained no scored dv points; skipped')
            continue
        dvs = np.array([p[0] for p in pts])
        epss = np.array([p[1] for p in pts])
        ns = np.array(ns)
        floor = data.get('negligible_dv_floor') or {}
        f_dv50 = data.get('dv50')
        f_dv90 = data.get('dv90')
        return (path, dvs, epss, ns, floor,
                None if f_dv50 is None else float(f_dv50),
                None if f_dv90 is None else float(f_dv90))
    return None


def crossing_eps(dv, eps, level):
    """Interpolate the dv at which the curve first crosses `level`."""
    if np.any(eps >= level):
        p = PchipInterpolator(np.log10(dv), eps)
        grid = np.logspace(np.log10(dv.min()), np.log10(dv.max()), 4001)
        sm = np.clip(p(np.log10(grid)), 0, 1)
        above = np.where(sm >= level)[0]
        if len(above):
            return float(grid[above[0]])
    return None


def extract_detected(payload):
    """Best-effort extraction of per-model expected-detected counts.

    Handles the eval_fleet_efficiency.py schema ({"summary": {"ordem": {...},
    "master": {...}}}) as well as top-level or flat-field variants.
    """
    out = {}
    if not isinstance(payload, dict):
        return out
    candidates = [payload]
    for section in ('summary', 'totals', 'results'):
        if isinstance(payload.get(section), dict):
            candidates.append(payload[section])
    for source in candidates:
        lowered = {str(k).lower(): v for k, v in source.items()}
        for model in ('ordem', 'master'):
            if model in out:
                continue
            v = lowered.get(model)
            if isinstance(v, dict):
                vlow = {str(k).lower().replace(' ', '_'): vv
                        for k, vv in v.items()}
                for key in ('total_detected', 'detected', 'expected_detected',
                            'n_detected', 'detected_expected',
                            'total_expected_detected'):
                    if key in vlow:
                        out[model] = float(vlow[key])
                        break
            elif isinstance(v, (int, float)):
                out[model] = float(v)
            if model not in out:
                for key in (f'{model}_detected', f'total_{model}_detected',
                            f'{model}_total_detected'):
                    if isinstance(lowered.get(key), (int, float)):
                        out[model] = float(lowered[key])
                        break
    if not out:
        print('  [fig3/4] could not auto-parse expected-detections file;'
              ' top-level keys:', list(payload.keys()))
    return out

# ============================================================
# Load the operative sweep
# ============================================================
sweep_path, sweep_dv, sweep_eps, file_dv50 = load_efficiency_curve()
dv50 = crossing_eps(sweep_dv, sweep_eps, 0.5) or DV50_FALLBACK
dv90 = crossing_eps(sweep_dv, sweep_eps, 0.9)

print(f'Sweep loaded from:  {sweep_path}')
print(f"  points: {len(sweep_dv)}, dv range {sweep_dv.min():.1e} - {sweep_dv.max():.2f} m/s")
print(f'  file dv50 field:   {file_dv50}   <-- CROSS-CHECK against 1.25152')
print(f'                     (if it differs, the field is stale; do not quote it)')
print(f'  interpolated dv50: {dv50:.5f} m/s')
print(f'  interpolated dv90: {dv90 if dv90 is None else round(dv90, 4)} m/s')

# ============================================================
# Load the real-TLE injection curve (if present)
# ============================================================
real = load_real_tle_curve()
if real is None:
    print('  [real-TLE] no injection file found in: '
          + ', '.join(REAL_TLE_FILES) + '  -> figs 5/7 SKIPPED')
else:
    (real_path, real_dv, real_eps, real_n, real_floor,
     real_file_dv50, real_file_dv90) = real
    real_dv50 = crossing_eps(real_dv, real_eps, 0.5)
    real_dv90 = crossing_eps(real_dv, real_eps, 0.9)
    print(f'Real-TLE injection loaded from: {real_path}')
    print(f"  points: {len(real_dv)}, dv range {real_dv.min():.1e} - {real_dv.max():.2f} m/s")
    print(f'  scored windows per dv: {real_n.min()}-{real_n.max()}')
    print(f'  file dv50/dv90:        {real_file_dv50} / {real_file_dv90}'
          '   <-- CROSS-CHECK against the interpolated values')
    print('  interpolated dv50:     '
          + (f'{real_dv50:.5f}' if real_dv50 is not None else 'not bracketed'))
    print('  interpolated dv90:     '
          + (f'{real_dv90:.5f}' if real_dv90 is not None else 'not bracketed'))
    fl = real_floor.get('epsilon')
    print('  negligible-dv floor:   '
          + ('%.4f%%' % (100 * fl) if fl is not None else 'N/A')
          + f" at dv={real_floor.get('dv', float('nan')):g} m/s")


# ============================================================
# FIGURE 1: Pipeline Schematic  (eta = 417.94)
# ============================================================
fig, ax = plt.subplots(figsize=(14, 7.5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7.5)
ax.axis('off')

ax.text(7, 7.0, 'IMPULSE Pipeline Process Flow', fontsize=20, fontweight='bold',
        ha='center', va='center', color='#111111')

def draw_box(ax, x, y, w, h, text, subtext=None, color='#E8F4F8',
             edgecolor='#2E86AB', fontsize=11, subfontsize=9.5):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                         facecolor=color, edgecolor=edgecolor, linewidth=1.8)
    ax.add_patch(box)
    if subtext:
        ax.text(x + w/2, y + h/2 + 0.18, text, fontsize=fontsize, fontweight='bold',
                ha='center', va='center', color='#111111')
        ax.text(x + w/2, y + h/2 - 0.22, subtext, fontsize=subfontsize,
                ha='center', va='center', color='#555555')
    else:
        ax.text(x + w/2, y + h/2, text, fontsize=fontsize, fontweight='bold',
                ha='center', va='center', color='#111111')

def draw_arrow(ax, x1, y1, x2, y2, color='#2E86AB'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=2, mutation_scale=15))

draw_box(ax, 0.6, 3.8, 2.5, 1.8, 'TLE History', '(3,325 objects)')
draw_box(ax, 3.7, 3.8, 2.5, 1.8, '90-day Windows', f'({TOTAL_WINDOWS:,} valid)')

central_box = FancyBboxPatch((6.8, 3.3), 3.4, 2.8,
                             boxstyle="round,pad=0.02,rounding_size=0.15",
                             facecolor='#FFF3E0', edgecolor='#E65100', linewidth=2.0)
ax.add_patch(central_box)
ax.text(8.5, 5.5, 'Forward EKF', fontsize=12, fontweight='bold', ha='center', va='center')
ax.text(8.5, 5.0, '+', fontsize=14, fontweight='bold', ha='center', va='center', color='#E65100')
ax.text(8.5, 4.5, 'Backward EKF', fontsize=12, fontweight='bold', ha='center', va='center')
ax.text(8.5, 3.8, r'$\rightarrow$ Fraser-Potter Smoother', fontsize=10.5,
        ha='center', va='center', color='#D84315', style='italic', fontweight='bold')

draw_box(ax, 10.8, 3.8, 2.6, 1.8, 'Peak Mahalanobis', 'Detection Analysis')

draw_box(ax, 6.8, 0.8, 3.4, 1.7, 'Fleet Aggregation',
         'Model Comparison (ORDEM vs MASTER)', color='#E8F5E9', edgecolor='#2E7D32')

draw_box(ax, 0.6, 0.8, 2.5, 1.7, 'Nonsingular State',
         r'$h = e \sin \omega$' + chr(10) + r'$k = e \cos \omega$',
         color='#F5F5F5', edgecolor='#9E9E9E', fontsize=10.5, subfontsize=10)

draw_box(ax, 10.8, 0.8, 2.6, 1.7, 'Detection Threshold',
         r'$\eta = 417.94$' + chr(10) + '(99.9th percentile, 0.1%/window)',
         color='#F5F5F5', edgecolor='#9E9E9E', fontsize=10.5, subfontsize=9.5)

draw_arrow(ax, 3.1, 4.7, 3.7, 4.7)
draw_arrow(ax, 6.2, 4.7, 6.8, 4.7)
draw_arrow(ax, 10.2, 4.7, 10.8, 4.7)
draw_arrow(ax, 8.5, 3.3, 8.5, 2.5, color='#2E7D32')

plt.tight_layout()
plt.savefig('figs/fig1_pipeline_schematic.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()


# ============================================================
# FIGURE 2: Detection Efficiency at the Design Threshold
# (single operative curve; no legacy fallback data)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(sweep_dv, sweep_eps, s=80, color='#E65100', zorder=5,
           label='Impulse sweep ($\eta = 417.94$)', marker='o',
           edgecolor='white', linewidth=0.5)

dv_fine = np.logspace(np.log10(sweep_dv.min()), np.log10(sweep_dv.max()), 400)
pchip = PchipInterpolator(np.log10(sweep_dv), sweep_eps)
ax.plot(dv_fine, np.clip(pchip(np.log10(dv_fine)), 0, 1),
        color='#E65100', lw=2.2, alpha=0.85)

ax.axhline(y=0.5, color='#A0A0A0', linestyle='--', lw=1.2, alpha=0.7)
ax.axhline(y=0.9, color='#A0A0A0', linestyle='--', lw=1.2, alpha=0.7)
ax.text(sweep_dv.min() * 1.3, 0.515, '50%', fontsize=9.5, color='#666666', fontweight='bold')
ax.text(sweep_dv.min() * 1.3, 0.915, '90%', fontsize=9.5, color='#666666', fontweight='bold')

ax.axvline(x=dv50, color='#E65100', linestyle=':', lw=1.5, alpha=0.7)
ax.annotate(rf'$\Delta v_{{50}} = {dv50:.2f}$ m/s',
            xy=(dv50, 0.5), xytext=(dv50 * 2.8, 0.66),
            fontsize=11, color='#E65100', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.2,
                            connectionstyle="arc3,rad=-0.15"))
if dv90 is not None:
    ax.axvline(x=dv90, color='#E65100', linestyle=':', lw=1.2, alpha=0.4)

# Measured negligible-dv floor of the sweep (anchor point)
floor_eps = sweep_eps[0]
ax.annotate(rf'negligible-$\Delta v$ floor: {floor_eps*100:.1f}%',
            xy=(sweep_dv[0], max(floor_eps, 0.01)),
            xytext=(sweep_dv[0] * 2.2, 0.10),
            fontsize=10, color='#666666', style='italic',
            arrowprops=dict(arrowstyle='->', color='#666666', lw=1.0))

ax.set_xscale('log')
ax.set_xlabel(r'$\Delta v$ (m/s)', fontsize=12)
ax.set_ylabel(r'Detection efficiency $\epsilon(\Delta v)$', fontsize=12)
ax.set_title(r'Detection Efficiency at the Design Threshold ($\eta = 417.94$)',
             fontsize=14, fontweight='bold', pad=12)
ax.set_xlim(sweep_dv.min() * 0.7, max(sweep_dv.max() * 1.5, 4))
ax.set_ylim(-0.02, 1.05)
ax.legend(loc='lower right', fontsize=10.5, framealpha=0.95, facecolor='#FFFFFF')
ax.grid(True, which="major", linestyle='-', linewidth=0.5, alpha=0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figs/fig2_efficiency_curves.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()


# ============================================================
# FIGURE 3: Observed vs Noise-Only vs Model Predictions
# (interim now; auto-upgrades when the two eval runs land)
# ============================================================
expected = {}
for path, band in EXPECTED_FILES.items():
    if os.path.exists(path):
        with open(path) as f:
            payload = json.load(f)
        det = extract_detected(payload)
        expected[band] = det
        print(f'  [fig3/4] {band}: parsed detected = {det}')

fig, ax = plt.subplots(figsize=(9.5, 6))

# Fixed current-number bars
fixed_labels = ['Observed' + chr(10) + '(fleet)',
                'Noise-only' + chr(10) + 'expected']
fixed_vals = [OBSERVED, NOISE_EXPECTED]
fixed_colors = ['#222222', '#A8DADC']

model_labels, model_vals, model_colors = [], [], []
palette = {'1 mm-1 cm': {'ordem': '#457B9D', 'master': '#8FBFD9'},
           '1-10 cm': {'ordem': '#2A9D8F', 'master': '#7FD8CC'}}
for band, det in expected.items():
    for model in ('ordem', 'master'):
        if model in det:
            model_labels.append(f'{model.upper()}' + chr(10) + f'{band}'
                                + chr(10) + '(noise + det.)')
            model_vals.append(NOISE_EXPECTED + det[model])
            model_colors.append(palette[band][model])

all_labels = fixed_labels + model_labels
all_vals = fixed_vals + model_vals
all_colors = fixed_colors + model_colors
x = np.arange(len(all_labels))

bars = ax.bar(x, all_vals, 0.55, color=all_colors, edgecolor='#1D3557', linewidth=1.0)
for xi, v in zip(x, all_vals):
    ax.text(xi, v + 0.25, f'{v:.2f}', ha='center', va='bottom',
            fontsize=10.5, fontweight='bold', color='#111111')

ax.set_xticks(x)
ax.set_xticklabels(all_labels, fontsize=9.5)
ax.set_ylabel(f'Trigger count ({TOTAL_WINDOWS:,} valid windows)', fontsize=12)
ax.set_title('Observed Triggers vs Noise-Only and Model Predictions '
             '($\eta = 417.94$)', fontsize=13, fontweight='bold', pad=15)
ax.set_ylim(0, max(all_vals) * 1.25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, axis='y', linestyle=':', alpha=0.5)

if not expected:
    ax.text(0.5, -0.18, 'Model-predicted bars pending the two expected-detections runs; '
            'interim version shows observed vs noise-only.',
            transform=ax.transAxes, ha='center', fontsize=9.5, style='italic', color='#777777')

plt.tight_layout()
plt.savefig('figs/fig3_trigger_decomposition.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()


# ============================================================
# FIGURE 4: Horizontal comparison (current numbers only)
# ============================================================
fig, ax = plt.subplots(figsize=(9.5, 4.5))

labels = ['Observed (fleet)', 'Noise-only expected']
rates_pct = [OBSERVED / TOTAL_WINDOWS * 100, NOISE_EXPECTED / TOTAL_WINDOWS * 100]
counts = [OBSERVED, NOISE_EXPECTED]
colors_list = ['#222222', '#457B9D']

for band, det in expected.items():
    for model in ('ordem', 'master'):
        if model in det:
            labels.append(f'{model.upper()} noise+det. ({band})')
            counts.append(NOISE_EXPECTED + det[model])
            rates_pct.append(counts[-1] / TOTAL_WINDOWS * 100)
            colors_list.append(palette[band][model])

y_pos = np.arange(len(labels))
bars = ax.barh(y_pos, counts, height=0.55, color=colors_list,
               edgecolor='#1D3557', linewidth=1.2)
for bar, cnt, rate in zip(bars, counts, rates_pct):
    ax.text(bar.get_width() + 0.25, bar.get_y() + bar.get_height() / 2,
            f'{cnt:.2f} ({rate:.3f}%)', va='center', ha='left',
            fontsize=10.5, fontweight='bold', color='#111111')

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=10.5, fontweight='bold')
ax.set_xlabel(f'Trigger count ({TOTAL_WINDOWS:,} valid windows)', fontsize=12)
ax.set_title('Observed vs Noise-Only vs Model Predictions at $\eta = 417.94$',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlim(0, max(counts) * 1.35)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, axis='x', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig('figs/fig4_horizontal_detection_rates.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()


# ============================================================
# FIGURE 5: Real-TLE Strike-Injection Efficiency (alone)
# ============================================================
if real is not None:
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(real_dv, real_eps, s=80, color='#2E86AB', zorder=5,
               label='Real-TLE strike injection ($\eta = 417.94$)',
               marker='s', edgecolor='white', linewidth=0.5)
    dv_fine_r = np.logspace(np.log10(real_dv.min()), np.log10(real_dv.max()), 400)
    pchip_r = PchipInterpolator(np.log10(real_dv), real_eps)
    ax.plot(dv_fine_r, np.clip(pchip_r(np.log10(dv_fine_r)), 0, 1),
            color='#2E86AB', lw=2.2, alpha=0.85)

    ax.axhline(y=0.5, color='#A0A0A0', linestyle='--', lw=1.2, alpha=0.7)
    ax.axhline(y=0.9, color='#A0A0A0', linestyle='--', lw=1.2, alpha=0.7)
    ax.text(real_dv.min() * 1.3, 0.515, '50%', fontsize=9.5,
            color='#666666', fontweight='bold')
    ax.text(real_dv.min() * 1.3, 0.915, '90%', fontsize=9.5,
            color='#666666', fontweight='bold')

    if real_dv50 is not None:
        ax.axvline(x=real_dv50, color='#2E86AB', linestyle=':', lw=1.5, alpha=0.7)
        ax.annotate(rf'$\Delta v_{{50}} = {real_dv50:.2f}$ m/s',
                    xy=(real_dv50, 0.5), xytext=(real_dv50 * 2.8, 0.66),
                    fontsize=11, color='#2E86AB', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=1.2,
                                    connectionstyle="arc3,rad=-0.15"))
    if real_dv90 is not None:
        ax.axvline(x=real_dv90, color='#2E86AB', linestyle=':', lw=1.2, alpha=0.4)
        ax.annotate(rf'$\Delta v_{{90}} = {real_dv90:.2f}$ m/s',
                    xy=(real_dv90, 0.9), xytext=(real_dv90 * 0.28, 0.78),
                    fontsize=10, color='#2E86AB',
                    arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=1.0,
                                    connectionstyle="arc3,rad=0.15"))

    fl_eps = real_floor.get('epsilon')
    if fl_eps is not None:
        floor_txt = rf'negligible-$\Delta v$ floor: {fl_eps*100:.1f}%'
        if real_floor.get('n_scored'):
            floor_txt += f'  ({real_floor["n_scored"]} scored)'
        ax.annotate(floor_txt,
                    xy=(real_dv[0], max(fl_eps, 0.01)),
                    xytext=(real_dv[0] * 2.2, 0.10),
                    fontsize=10, color='#666666', style='italic',
                    arrowprops=dict(arrowstyle='->', color='#666666', lw=1.0))

    ax.set_xscale('log')
    ax.set_xlabel(r'$\Delta v$ (m/s)', fontsize=12)
    ax.set_ylabel(r'Detection efficiency $\epsilon(\Delta v)$', fontsize=12)
    n_lo, n_hi = int(real_n.min()), int(real_n.max())
    sub = f'Certified on real catalog histories ({n_lo}-{n_hi} scored windows per dv)'
    ax.set_title(r'Real-TLE Strike-Injection Efficiency ($\eta = 417.94$)'
                 + chr(10) + sub,
                 fontsize=13.5, fontweight='bold', pad=12)
    ax.set_xlim(real_dv.min() * 0.7, max(real_dv.max() * 1.5, 4))
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc='lower right', fontsize=10.5, framealpha=0.95, facecolor='#FFFFFF')
    ax.grid(True, which="major", linestyle='-', linewidth=0.5, alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('figs/fig5_real_tle_efficiency.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


# ============================================================
# FIGURE 6: Synthetic Impulse Sweep Efficiency (alone)
# (same styling as fig5, so synthetic and real-TLE read as a pair)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(sweep_dv, sweep_eps, s=80, color='#E65100', zorder=5,
           label='Synthetic impulse sweep ($\eta = 417.94$)',
           marker='o', edgecolor='white', linewidth=0.5)

dv_fine_s = np.logspace(np.log10(sweep_dv.min()), np.log10(sweep_dv.max()), 400)
ax.plot(dv_fine_s, np.clip(pchip(np.log10(dv_fine_s)), 0, 1),
        color='#E65100', lw=2.2, alpha=0.85)

ax.axhline(y=0.5, color='#A0A0A0', linestyle='--', lw=1.2, alpha=0.7)
ax.axhline(y=0.9, color='#A0A0A0', linestyle='--', lw=1.2, alpha=0.7)
ax.text(sweep_dv.min() * 1.3, 0.515, '50%', fontsize=9.5, color='#666666', fontweight='bold')
ax.text(sweep_dv.min() * 1.3, 0.915, '90%', fontsize=9.5, color='#666666', fontweight='bold')

ax.axvline(x=dv50, color='#E65100', linestyle=':', lw=1.5, alpha=0.7)
ax.annotate(rf'$\Delta v_{{50}} = {dv50:.2f}$ m/s',
            xy=(dv50, 0.5), xytext=(dv50 * 2.8, 0.66),
            fontsize=11, color='#E65100', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.2,
                            connectionstyle="arc3,rad=-0.15"))
if dv90 is not None:
    ax.axvline(x=dv90, color='#E65100', linestyle=':', lw=1.2, alpha=0.4)
    ax.annotate(rf'$\Delta v_{{90}} = {dv90:.2f}$ m/s',
                xy=(dv90, 0.9), xytext=(dv90 * 0.28, 0.78),
                fontsize=10, color='#E65100',
                arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.0,
                                connectionstyle="arc3,rad=0.15"))

floor_eps_s = sweep_eps[0]
ax.annotate(rf'negligible-$\Delta v$ floor: {floor_eps_s*100:.1f}%',
            xy=(sweep_dv[0], max(floor_eps_s, 0.01)),
            xytext=(sweep_dv[0] * 2.2, 0.10),
            fontsize=10, color='#666666', style='italic',
            arrowprops=dict(arrowstyle='->', color='#666666', lw=1.0))

ax.set_xscale('log')
ax.set_xlabel(r'$\Delta v$ (m/s)', fontsize=12)
ax.set_ylabel(r'Detection efficiency $\epsilon(\Delta v)$', fontsize=12)
ax.set_title(r'Synthetic Impulse-Sweep Efficiency ($\eta = 417.94$)'
             + chr(10) + f'Operative gate-4 sweep ({len(sweep_dv)} dv points)',
             fontsize=13.5, fontweight='bold', pad=12)
ax.set_xlim(sweep_dv.min() * 0.7, max(sweep_dv.max() * 1.5, 4))
ax.set_ylim(-0.02, 1.05)
ax.legend(loc='lower right', fontsize=10.5, framealpha=0.95, facecolor='#FFFFFF')
ax.grid(True, which="major", linestyle='-', linewidth=0.5, alpha=0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figs/fig6_synthetic_efficiency.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()


# ============================================================
# FIGURE 7: Synthetic Sweep vs Real-TLE Injection (both curves)
# ============================================================
if real is not None:
    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Synthetic sweep (orange)
    ax.scatter(sweep_dv, sweep_eps, s=70, color='#E65100', zorder=5,
               marker='o', edgecolor='white', linewidth=0.5,
               label='Synthetic impulse sweep')
    ax.plot(dv_fine_s, np.clip(pchip(np.log10(dv_fine_s)), 0, 1),
            color='#E65100', lw=2.0, alpha=0.8)

    # Real-TLE injection (blue)
    ax.scatter(real_dv, real_eps, s=70, color='#2E86AB', zorder=6,
               marker='s', edgecolor='white', linewidth=0.5,
               label='Real-TLE strike injection')
    ax.plot(dv_fine_r, np.clip(pchip_r(np.log10(dv_fine_r)), 0, 1),
            color='#2E86AB', lw=2.0, alpha=0.8, linestyle='--')

    ax.axhline(y=0.5, color='#A0A0A0', linestyle='--', lw=1.2, alpha=0.7)
    ax.axhline(y=0.9, color='#A0A0A0', linestyle='--', lw=1.2, alpha=0.7)
    ax.text(sweep_dv.min() * 1.3, 0.515, '50%', fontsize=9.5,
            color='#666666', fontweight='bold')
    ax.text(sweep_dv.min() * 1.3, 0.915, '90%', fontsize=9.5,
            color='#666666', fontweight='bold')

    if dv50 is not None:
        ax.axvline(x=dv50, color='#E65100', linestyle=':', lw=1.5, alpha=0.7)
    if real_dv50 is not None:
        ax.axvline(x=real_dv50, color='#2E86AB', linestyle=':', lw=1.5, alpha=0.7)

    # Summary box: both dv50 / dv90 in one place
    lines = []
    if dv50 is not None:
        line = f'Synthetic: dv50 = {dv50:.3f} m/s'
        if dv90 is not None:
            line += f',  dv90 = {dv90:.3f} m/s'
        lines.append(line)
    if real_dv50 is not None:
        line = f'Real-TLE:  dv50 = {real_dv50:.3f} m/s'
        if real_dv90 is not None:
            line += f',  dv90 = {real_dv90:.3f} m/s'
        lines.append(line)
    if lines:
        ax.text(0.97, 0.05, chr(10).join(lines), transform=ax.transAxes,
                fontsize=10.5, va='bottom', ha='right', family='sans-serif',
                bbox=dict(boxstyle='round,pad=0.45', facecolor='#FFFFFF',
                          edgecolor='#CCCCCC', alpha=0.92))

    ax.set_xscale('log')
    ax.set_xlabel(r'$\Delta v$ (m/s)', fontsize=12)
    ax.set_ylabel(r'Detection efficiency $\epsilon(\Delta v)$', fontsize=12)
    ax.set_title(r'Impulse Sensitivity: Synthetic vs Real-TLE Certification'
                 + chr(10) + r'($\eta = 417.94$, shared windows / covariance / threshold)',
                 fontsize=13.5, fontweight='bold', pad=12)
    ax.set_xlim(min(sweep_dv.min(), real_dv.min()) * 0.7,
                max(sweep_dv.max(), real_dv.max()) * 1.5)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc='lower right', fontsize=10.5, framealpha=0.95, facecolor='#FFFFFF')
    ax.grid(True, which="major", linestyle='-', linewidth=0.5, alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('figs/fig7_efficiency_synthetic_vs_real.png', dpi=300,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()


print('All figures rendered.')
print('  fig1: pipeline schematic (eta = 417.94, 15,760 valid windows)')
print(f'  fig2: efficiency curve (dv50 = {dv50:.5f} m/s'
      + (f', dv90 = {dv90:.4f} m/s' if dv90 is not None else '') + ')')
print('  fig3/fig4: ' + ('full version with model predictions'
      if expected else 'interim (observed vs noise-only); '
      'will auto-upgrade when the two expected-detections JSONs land'))
print('  fig6: synthetic-only efficiency curve (companion to fig5)')
if real is not None:
    s50 = f'dv50 = {real_dv50:.5f} m/s' if real_dv50 is not None else 'dv50 not bracketed'
    s90 = f', dv90 = {real_dv90:.4f} m/s' if real_dv90 is not None else ''
    print(f'  fig5: real-TLE injection curve ({s50}{s90})')
    print('  fig7: synthetic vs real-TLE overlay')
else:
    print('  fig5/fig7: skipped (no real-TLE injection file)')