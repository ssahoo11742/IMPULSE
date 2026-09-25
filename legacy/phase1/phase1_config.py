"""Phase 1 — Signal Existence Audit: Configuration"""
import numpy as np
from scipy.stats import qmc

PARAMETER_BOUNDS = {
    'log10_rho_debris': [-15.0, -8.0],
    'Cd_base': [1.5, 3.0],
    'mass_kg': [50.0, 2000.0],
    'area_m2': [0.5, 20.0],
    'f107_base': [70.0, 250.0],
    'storm_rate_factor': [0.5, 2.0],
    'altitude_km': [400.0, 1200.0],
    'inclination_deg': [0.0, 98.0],
    'duration_days': [30.0, 730.0],
}
PARAMETER_NAMES = list(PARAMETER_BOUNDS.keys())
N_PARAMS = len(PARAMETER_NAMES)
DT_S = 86400.0
EPOCH_JD = 2460000.5
N_ENSEMBLE = 8

def sample_sobol(n_samples, n_params=None, seed=42):
    if n_params is None: n_params = N_PARAMS
    return qmc.Sobol(d=n_params, scramble=True, seed=seed).random(n=n_samples)

def sample_lhs(n_samples, n_params=None, seed=42):
    if n_params is None: n_params = N_PARAMS
    return qmc.LatinHypercube(d=n_params, seed=seed).random(n=n_samples)

def map_to_bounds(unit_samples, bounds_dict=None, param_names=None):
    if bounds_dict is None: bounds_dict = PARAMETER_BOUNDS
    if param_names is None: param_names = PARAMETER_NAMES
    n = len(unit_samples)
    samples = np.zeros((n, len(param_names)))
    for i, name in enumerate(param_names):
        low, high = bounds_dict[name]
        samples[:, i] = low + unit_samples[:, i] * (high - low)
    return samples

def get_parameter_dict(sample_row, param_names=None):
    if param_names is None: param_names = PARAMETER_NAMES
    return {name: float(sample_row[i]) for i, name in enumerate(param_names)}

def deterministic_seed(params_dict, replicate_id):
    key = tuple(sorted((k, round(v, 6)) for k, v in params_dict.items())) + (replicate_id,)
    return int(hash(key) % (2**32))

