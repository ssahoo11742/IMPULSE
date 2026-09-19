"""Generate fig_241_impactor_size.{pdf,png} — impulse vs. impactor size mapping.

Standalone physics plot used in §2.4.3: for a perfectly inelastic impact
(beta = 1), an impactor of mass m at relative velocity v_rel delivers

    Delta_v = beta * m * v_rel / M_sc

so the diameter of the impactor corresponding to a given impulse is

    d = 2 * (3 * beta * Delta_v * M_sc / (4 * pi * rho * v_rel))^(1/3)

Vertical lines mark the fitted Gate 3b sensitivity, Delta_v_50 = 0.0385 m/s
and Delta_v_90 = 0.0683 m/s (real_tle_strike_injection.py results).

Usage:
    python make_fig_241_impactor_size.py
"""

import numpy as np
import matplotlib.pyplot as plt

RHO = 2700.0                      # kg/m^3, solid density of a mm-cm class impactor
VREL = 10e3                       # m/s, typical LEO collision velocity
DV50, DV90 = 0.0385, 0.0683       # m/s, fitted real-TLE sensitivity (Gate 3b)


def diameter(dv, mass, beta=1.0):
    """Impactor diameter [m] delivering impulse dv [m/s] on a spacecraft of
    mass `mass` [kg]; beta = momentum enhancement factor."""
    m_imp = beta * dv * mass / VREL
    return 2.0 * (3.0 * m_imp / (4.0 * np.pi * RHO)) ** (1.0 / 3.0)


def main():
    dv = np.logspace(-5, 0, 100)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for mass, ls in [(10, ':'), (100, '-'), (1000, '--')]:
        ax.loglog(dv * 1e3, diameter(dv, mass) * 1e3, ls, lw=1.8, color='k',
                  label=fr'$M = {mass}$ kg, $\beta = 1$')
    ax.loglog(dv * 1e3, diameter(dv, 100, beta=2) * 1e3, '-', lw=1.2,
              color='gray', label=r'$M = 100$ kg, $\beta = 2$')

    for d0, lab in [(DV50, r'$\Delta v_{50} = 0.0385$ m/s'),
                    (DV90, r'$\Delta v_{90} = 0.0683$ m/s')]:
        ax.axvline(d0 * 1e3, color='r', alpha=0.5, lw=1)
        ax.annotate(lab, xy=(d0 * 1e3 * 1.1, 8), rotation=90, fontsize=8,
                    va='top', color='r')

    ax.set_xlabel(r'impulse $\Delta v$  [mm/s]')
    ax.set_ylabel('impactor diameter [mm]')
    ax.set_title(r'Impulse delivered vs. impactor size ($v_{rel} = 10$ km/s, '
                 r'$\rho = 2700$ kg/m$^3$)', fontsize=10)
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig('figs/fig_241_impactor_size.pdf')
    fig.savefig('figs/fig_241_impactor_size.png', dpi=200)
    plt.show()

    print("diameter at dv50, 100 kg, beta=1: %.1f mm" % (diameter(DV50, 100) * 1e3))
    print("diameter at 0.1 m/s, 100 kg, beta=1: %.1f mm" % (diameter(0.1, 100) * 1e3))


if __name__ == '__main__':
    main()