"""Test statistics extracted from EKF + smoother output.

All functions operate on lists of 9-state vectors (and associated covariances)
produced by the forward/backward/smoother pipeline.
"""

import numpy as np
from typing import List, Tuple, Optional


def compute_smoothed_accel_peak(
    sm_states: List[np.ndarray],
    times: List[float],
    strike_window: Optional[Tuple[float, float]] = None,
) -> dict:
    """Return peak magnitude of the smoothed RSW acceleration.

    Parameters
    ----------
    sm_states : list of np.ndarray, shape (9,)
        Smoothed state vectors.
    times : list of float
        Epoch times [s], same length as sm_states.
    strike_window : (t_start, t_end) or None
        If provided, restrict the search to this window.

    Returns
    -------
    dict with keys:
        peak_norm   : max |w_S|  (Euclidean norm of [w_R, w_S, w_W])
        peak_time   : epoch of the peak
        peak_index  : index in the arrays
        mean_offpeak: mean |w_S| outside the peak epoch (noise floor proxy)
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

    # noise floor: mean of everything except the peak epoch
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

    Bennett Eq. 19–20:
        X_M = X_F - X_B_bar
        D_MH = sqrt( X_M^T * P_S^{-1} * X_M )

    Parameters
    ----------
    state_mask : np.ndarray of bool, shape (9,)
        If provided, restrict to a subspace (e.g. only a & M for along-track).
        Default uses all 9 states.
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

        # Regularised inverse
        eigvals, eigvecs = np.linalg.eigh(P_sub)
        eigvals = np.maximum(eigvals, 1.0e-12)
        P_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T

        d[i] = np.sqrt(max(0.0, dx_sub @ P_inv @ dx_sub))

    return d


def compute_mcreynolds(
    fwd_states: List[np.ndarray],
    fwd_covs: List[np.ndarray],
    sm_states: List[np.ndarray],
    sm_covs: List[np.ndarray],
) -> dict:
    """McReynold's filter–smoother consistency test.

    Per-element (Eq. 26):
        R_m,j = |X_F,j - X_S,j| / sqrt(P_F,jj - P_S,jj)

    Scalar (Eq. 27, dimensionally cleaned up):
        R_scalar = sqrt( sum_j R_m,j^2 )

    Returns per-epoch arrays.
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