"""Numerically robust eigendecomposition helpers for ill-conditioned covariances."""
import numpy as np


def _eigh_with_retry(M, max_retries=4):
    """Try eigh, adding increasing relative jitter on failure.
    Returns (eigvals, eigvecs), or None if all retries fail."""
    scale = float(np.mean(np.abs(np.diag(M)))) or 1.0
    jitter = 0.0
    for attempt in range(max_retries):
        try:
            M_try = M + jitter * np.eye(M.shape[0]) if jitter > 0 else M
            return np.linalg.eigh(M_try)
        except np.linalg.LinAlgError:
            jitter = scale * 1.0e-9 if jitter == 0 else jitter * 10.0
    return None


def safe_eigh_floor(M, floor=1.0e-12):
    """Symmetric M with eigenvalues floored at `floor`. Diagonal fallback on total failure."""
    result = _eigh_with_retry(M)
    if result is None:
        return np.eye(M.shape[0]) * floor
    eigvals, eigvecs = result
    eigvals = np.maximum(eigvals, floor)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def safe_eigh_inverse(M, floor=1.0e-12):
    """Regularized inverse of symmetric M. Falls back to pinv; raises if even that fails."""
    result = _eigh_with_retry(M)
    if result is None:
        try:
            return np.linalg.pinv(M, rcond=floor)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(
                "Covariance matrix is numerically degenerate — "
                "likely diverged (tau, q, duration). Skip this candidate."
            ) from e
    eigvals, eigvecs = result
    eigvals = np.maximum(eigvals, floor)
    return eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T