"""Shared numerically-robust eigendecomposition helpers.

np.linalg.eigh can raise LinAlgError("Eigenvalues did not converge") for
severely ill-conditioned symmetric matrices. This shows up in this project
at long durations combined with extreme tau/q choices, where covariance
matrices can span many orders of magnitude between their largest and
smallest eigenvalues (a direct consequence of the huge a<->w_S sensitivity
already documented elsewhere in this codebase - da/dt ~ 1900x per unit
w_S). Rather than let an entire multi-year run crash partway through
(which is exactly what was blocking the impulse-detection tuning at
longer, more realistic fleet durations), these helpers add a small jitter
(relative to the matrix's own scale) and retry a few times before falling
back to an SVD-based pseudo-inverse, which tolerates ill-conditioning that
eigh cannot.
"""
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
    """Return a regularized version of symmetric M (eigenvalues floored
    at `floor`). Falls back to a floor-only diagonal in the rare case
    where even jittered eigh repeatedly fails, rather than crashing."""
    result = _eigh_with_retry(M)
    if result is None:
        return np.eye(M.shape[0]) * floor
    eigvals, eigvecs = result
    eigvals = np.maximum(eigvals, floor)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def safe_eigh_inverse(M, floor=1.0e-12):
    """Return a regularized inverse of symmetric M. Falls back to
    numpy.linalg.pinv (SVD-based, more robust to ill-conditioning than
    eigh) if eigh fails even with jitter retries.

    If even pinv fails, the matrix isn't just ill-conditioned - it has
    genuinely diverged (usually NaN/Inf from an unstable (tau, q, duration)
    combination). Silently returning a fallback value here would produce a
    meaningless-but-plausible-looking number. Raise instead, so callers
    doing a grid search (tune_impulse_filter.py, find_dv_threshold.py) can
    catch this and skip that candidate/trial rather than trusting it."""
    result = _eigh_with_retry(M)
    if result is None:
        try:
            return np.linalg.pinv(M, rcond=floor)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(
                "Covariance matrix is numerically degenerate (not just "
                "ill-conditioned) - this (tau, q, duration) combination has "
                "likely diverged. Skip this candidate/trial rather than "
                "trusting a forced result."
            ) from e
    eigvals, eigvecs = result
    eigvals = np.maximum(eigvals, floor)
    return eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T