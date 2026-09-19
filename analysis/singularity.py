"""Generate fig_222_singularity_hk.{pdf,png} — Gauss-VOP singularity diagnostic.

Replicates the coefficient algebra of compute_B_matrix() in
phase2_filter/dynamics.py (the (h,k) state-representation fix) and plots the
classical d(omega)/dt coefficients — which diverge as 1/e — against the
non-singular (h,k) coefficients, which stay bounded all the way to e = 0.

Physical constants from constants/constants.py (WGS84 / EGM96).
Reference orbit: a = 7000 km (~600 km altitude), i = 98 deg (SSO).

Usage:
    python make_fig_222_singularity.py
"""

import math

import numpy as np
import matplotlib.pyplot as plt

# --- constants (constants/constants.py) ---
MU = 3.986004418e14                    # m^3/s^2, WGS84 Earth gravitational parameter

# --- reference orbit geometry ---
A_REF = 7.0e6                          # m, semi-major axis
INC = math.radians(98.0)               # rad, SSO reference inclination
ARGP = math.radians(45.0)              # rad, argument of perigee
NU = math.radians(30.0)                # rad, true anomaly (representative)


def coefficients(e: float):
    """|coefficients| of [w_R, w_S] in the d(argp)/dt, dh/dt, dk/dt rates.

    Mirrors dynamics.py::compute_B_matrix. The classical argument-of-perigee
    coefficients are A_argp/e and B_argp/e (singular); the (h,k) coefficients
    are the bounded chain-rule combinations (non-singular).
    """
    p = A_REF * (1.0 - e * e)
    r = p / (1.0 + e * math.cos(NU))
    h_ang = math.sqrt(MU * p)          # orbital angular momentum per unit mass
    cn, sn = math.cos(NU), math.sin(NU)

    de_R = p * sn / h_ang              # de/dt coefficient of w_R
    de_S = ((p + r) * cn + r * e) / h_ang   # de/dt coefficient of w_S

    A = -p * cn / h_ang                # d(argp)/dt coefficient of w_R (unnormalized)
    B = (p + r) * sn / h_ang           # d(argp)/dt coefficient of w_S (unnormalized)

    sing_R = abs(A / e)                # classical row, w_R  ~ 1/e
    sing_S = abs(B / e)                # classical row, w_S  ~ 1/e
    dh_R = abs(de_R * math.sin(ARGP) + math.cos(ARGP) * A)
    dh_S = abs(de_S * math.sin(ARGP) + math.cos(ARGP) * B)
    dk_R = abs(de_R * math.cos(ARGP) - math.sin(ARGP) * A)
    dk_S = abs(de_S * math.cos(ARGP) - math.sin(ARGP) * B)
    return sing_R, sing_S, dh_R, dh_S, dk_R, dk_S


def main():
    es = np.logspace(-8, -0.7, 200)
    cols = np.array([coefficients(e) for e in es])   # sR, sS, dhR, dhS, dkR, dkS

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True)
    panels = [(0, 2, 4, 'radial acceleration $w_R$'),
              (1, 3, 5, 'along-track acceleration $w_S$')]
    
    for ax, (idx_s, idx_h, idx_k, ttl) in zip(axes, panels):
        ax.loglog(es, cols[:, idx_s], color='#D62728', ls='--', lw=2,
                  label=r'$d\omega/dt$ (classical, $\propto 1/e$)')
        ax.loglog(es, cols[:, idx_h], color='#1F77B4', ls='-', lw=1.8,
                  label=r'$dh/dt$ (non-singular)')
        ax.loglog(es, cols[:, idx_k], color='#2CA02C', ls='-', lw=1.8,
                  label=r'$dk/dt$ (non-singular)')
        
        # 1. Expand y-limits so legend & bottom lines don't collide
        ymin = cols[:, [idx_s, idx_h, idx_k]].min()
        ymax = cols[:, idx_s].max()
        ax.set_ylim(ymin * 0.1, ymax * 10)

        # 2. Add reference lines with offset text boxes to prevent line overlap
        for e0, lab in ((1e-5, r'$e=10^{-5}$'), (1e-1, r'$e=0.1$')):
            ax.axvline(e0, color='gray', linestyle=':', alpha=0.6, lw=1)
            ax.annotate(
                lab, xy=(e0, ymax * 0.2), xytext=(e0 * 1.3, ymax * 0.2),
                fontsize=8, rotation=90, va='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8)
            )

        ax.set_xlabel('eccentricity $e$')
        ax.set_title(ttl, fontsize=10.5, pad=8)
        
        # 3. Clean up grid (major gridlines only to prevent background clutter)
        ax.grid(True, which='major', linestyle='-', alpha=0.3)
        ax.grid(True, which='minor', linestyle=':', alpha=0.15)
        
        # 4. Position legend outside plot area (upper right)
        ax.legend(fontsize=8, loc='upper right', framealpha=0.9)

    axes[0].set_ylabel('coefficient magnitude [s]')
    fig.suptitle('Gauss-VOP coefficients vs. eccentricity (a = 7000 km, i = 98°)', fontsize=11)
    
    fig.tight_layout()
    fig.savefig('figs/fig_222_singularity_hk.pdf')
    fig.savefig('figs/fig_222_singularity_hk.png', dpi=200)
    plt.show()


if __name__ == '__main__':
    main()