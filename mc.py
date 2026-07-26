import numpy as np
from MC.analysis import run_morris_screening, run_sobol_sensitivity_analysis, run_morris_screening_multi
from propagator.orbital import MeanElements  # Replace with your actual import path

if __name__ == "__main__":
    # Define Parameter Space
    problem = {
        'num_vars': 5,
        'names': ['rho_debris', 'Cd_base', 'area', 'mass', 'f107_base'],
        'bounds': [
            [1e-9, 1e-6],    # rho_debris [fragments/m^3]
            [1.8, 3.0],      # Cd_base
            [0.1, 10.0],     # area [m^2]
            [10.0, 500.0],   # mass [kg]
            [70.0, 250.0]    # f107_base [SFU]
        ]
    }

    # Nominal 600km orbit
    el0 = MeanElements(a=6978137.0, ecc=0.001, inc=np.radians(53.0), raan=0, argp=0, M=0)
    epoch_jd = 2460000.5

    # 1. TIER 1: Rapid Morris Test (N=10 -> 60 total trials, completes in ~5s)
    morris_results = run_morris_screening_multi(
        problem=problem,
        el0=el0,
        epoch_jd=epoch_jd,
        duration_s=86400 * 365,  # 7 days
        dt_s=86400,
        N=10,
        num_workers=8
    )

    # 2. TIER 2: Sobol Variance Decomposition
    # sobol_results = run_sobol_sensitivity_analysis(
    #     problem=problem,
    #     el0=el0,
    #     epoch_jd=epoch_jd,
    #     duration_s=86400 * 14, # 14 days
    #     dt_s=86400,
    #     n_samples=32,          # 32 * (2*5 + 2) = 384 points (Fast local test)
    #     n_replicates=2,
    #     num_workers=8
    # )