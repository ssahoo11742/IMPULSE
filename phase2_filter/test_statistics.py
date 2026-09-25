"""Test statistics from EKF + smoother output.

All of these take lists of 9-state vectors (and covariances) from the
forward / backward / smoother pipeline.
"""

from .linalg_utils import safe_eigh_inverse
import numpy as np
from typing import List, Tuple, Optional


def compute_smoothed_accel_peak(
    sm_states: List[np.ndarray],
    times: List[float],
    strike_window: Optional[Tuple[float, float]] = None,
) -> dict:
    """Peak magnitude of the smoothed RSW acceleration.

    If strike_window is given, only look inside that interval.
    Returns peak_norm, peak_time, peak_index, mean_offpeak, and snr.
    """
    assert len(sm_states) == len(times)
    w_norms = np.array([np.linalg.norm(x[6:9]) for x in sm_states])

    if strike_window is not None:
        t0, t1 = strike_window
        mask = np.array([(t0 <= t <= t1) for t in times])
        if not np.any(mask):
            mask = np.ones_like(mask, dtype=bool)
    else:
        mask = np.ones(len(times), dtype=bool)

    idx_peak = int(np.argmax(w_norms * mask))
    peak_norm = float(w_norms[idx_peak])
    peak_time = times[idx_peak]

    # noise floor: everything except the peak sample
    offpeak = np.delete(w_norms, idx_peak)
    mean_offpeak = float(np.mean(offpeak)) if len(offpeak) > 0 else 0.0

    return {
        "peak_norm": peak_norm,
        "peak_time": peak_time,
        "peak_index": idx_peak,
        "mean_offpeak": mean_offpeak,
        "snr": peak_norm / (mean_offpeak + 1.0e-12),
    }


def compute_mahalanobis_distance(
    fwd_states: List[np.ndarray],
    fwd_covs: List[np.ndarray],
    bwd_states: List[np.ndarray],
    bwd_covs: List[np.ndarray],
    sm_covs: List[np.ndarray],
    state_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Mahalanobis distance between forward and backward states.

    dx = x_f - x_b, then D = sqrt(dx^T P_s^{-1} dx).
    state_mask can restrict to a subspace (e.g. only a few components).
    """
    n = len(fwd_states)
    d = np.zeros(n)

    if state_mask is None:
        state_mask = np.ones(9, dtype=bool)

    for i in range(n):
        dx = fwd_states[i] - bwd_states[i]
        dx_sub = dx[state_mask]
        P_s = sm_covs[i]
        P_sub = P_s[np.ix_(state_mask, state_mask)]

        P_inv = safe_eigh_inverse(P_sub)
        d[i] = np.sqrt(max(0.0, dx_sub @ P_inv @ dx_sub))

    return d


def compute_mcreynolds(
    fwd_states: List[np.ndarray],
    fwd_covs: List[np.ndarray],
    sm_states: List[np.ndarray],
    sm_covs: List[np.ndarray],
) -> dict:
    """McReynolds filter-smoother consistency check.

    Per component: R_j = |x_f,j - x_s,j| / sqrt(P_f,jj - P_s,jj)
    Scalar version is just the L2 norm of those.
    """
    n = len(fwd_states)
    n_states = fwd_states[0].shape[0]
    R_per_state = np.zeros((n, n_states))
    R_scalar = np.zeros(n)

    for i in range(n):
        x_d = fwd_states[i] - sm_states[i]
        p_d_diag = np.diag(fwd_covs[i]) - np.diag(sm_covs[i])
        p_d_diag = np.maximum(p_d_diag, 1.0e-18)
        sigma_d = np.sqrt(p_d_diag)

        R_per_state[i, :] = np.abs(x_d) / sigma_d
        R_scalar[i] = np.linalg.norm(R_per_state[i, :])

    return {
        "per_state": R_per_state,
        "scalar": R_scalar,
    }