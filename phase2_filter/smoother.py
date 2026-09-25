"""Fraser-Potter fixed-interval smoother for the 9-state augmented EKF.

Combines forward a-posteriori with backward a-priori estimates
(Bennett Sec. III.C).
"""

from .linalg_utils import safe_eigh_inverse
import numpy as np
from typing import List, Tuple


def fraser_potter_smoother(
    fwd_states: List[np.ndarray],
    fwd_covs: List[np.ndarray],
    bwd_states: List[np.ndarray],
    bwd_covs: List[np.ndarray],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Fuse forward and backward filter outputs into smoothed estimates.

    fwd_* are the forward-filter a-posteriori values (after measurement update).
    bwd_* are the backward-filter a-priori values (after prediction, before
    the backward update). That way each measurement is only counted once.
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

        # W = P_b (P_f + P_b)^{-1}
        P_sum = P_f + P_b
        if not np.all(np.isfinite(P_sum)):
            return fwd_states, fwd_covs
        P_sum_inv = safe_eigh_inverse(P_sum)

        W = P_b @ P_sum_inv

        x_s = W @ x_f + (I - W) @ x_b
        P_s = W @ P_f @ W.T + (I - W) @ P_b @ (I - W).T

        sm_states.append(x_s)
        sm_covs.append(P_s)

    return sm_states, sm_covs