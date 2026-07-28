"""Fraser–Potter fixed-interval smoother for the 9-state augmented EKF.

Fuses forward a posteriori with backward a priori, exactly as Bennett
Sec. III.C describes.
"""

import numpy as np
from typing import List, Tuple


def fraser_potter_smoother(
    fwd_states: List[np.ndarray],
    fwd_covs: List[np.ndarray],
    bwd_states: List[np.ndarray],
    bwd_covs: List[np.ndarray],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Fuse forward and backward filter outputs into smoothed estimates.

    Parameters
    ----------
    fwd_states, fwd_covs
        Forward-filter a *posteriori* state and covariance at each epoch
        (i.e. after the measurement update).
    bwd_states, bwd_covs
        Backward-filter a *priori* state and covariance at each epoch
        (i.e. after the backward prediction, before the backward measurement
        update).  This ensures the measurement at epoch k is included
        exactly once — via the forward update.

    Returns
    -------
    sm_states, sm_covs : lists of np.ndarray
        Smoothed state and covariance at each epoch.
    """
    assert len(fwd_states) == len(bwd_states)
    n = len(fwd_states)
    sm_states: List[np.ndarray] = []
    sm_covs: List[np.ndarray] = []

    I = np.eye(9)

    for i in range(n):
        x_f = fwd_states[i]
        P_f = fwd_covs[i]
        x_b = bwd_states[i]
        P_b = bwd_covs[i]

        # Weighting matrix  W = P_b * (P_f + P_b)^{-1}
        P_sum = P_f + P_b
        eigvals, eigvecs = np.linalg.eigh(P_sum)
        eigvals = np.maximum(eigvals, 1.0e-12)
        P_sum_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T

        W = P_b @ P_sum_inv

        # Smoothed state and covariance (Bennett Eq. 21–23)
        x_s = W @ x_f + (I - W) @ x_b
        P_s = W @ P_f @ W.T + (I - W) @ P_b @ (I - W).T

        sm_states.append(x_s)
        sm_covs.append(P_s)

    return sm_states, sm_covs