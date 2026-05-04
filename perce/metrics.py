"""
Evaluation metrics for multivariate time series counterfactual explanations.

Adapts the standard counterfactual metrics from the CEval toolkit to
multivariate time series, as described in Section III-D of the PerCE paper.

Metrics:
  - Proximity  : DTW distance between original and CF, normalised by C*T
  - Sparsity   : fraction of segments NOT modified
  - Validity   : binary — did the CF achieve the desired class?
  - Diversity  : average pairwise DTW distance across a set of CFs
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .importance import _segment_boundaries
from .neighbors import _multivariate_dtw


# Proximity
def proximity(
    X_original: np.ndarray,
    X_cf: np.ndarray,
    dtw_window: float = 0.1,
) -> float:
    """DTW distance between original and counterfactual, normalised by C*T.

    Lower is better — counterfactual is closer to the original.

    Parameters
    ----------
    X_original, X_cf : np.ndarray, shape (C, T)
    dtw_window : float
        Sakoe-Chiba band as fraction of T.

    Returns
    -------
    float
    """
    C, T = X_original.shape
    window = max(1, int(dtw_window * T))
    dtw = _multivariate_dtw(X_original, X_cf, window=window)
    return dtw / (C * T)


# ──────────────────────────────────────────────────────────────────────
# Sparsity
# ──────────────────────────────────────────────────────────────────────


def sparsity(
    X_original: np.ndarray,
    X_cf: np.ndarray,
    n_segments: int = 10,
    tol: float = 1e-8,
) -> float:
    """Fraction of temporal segments that were NOT modified.

    Sparsity = 1 - (N_modified_segments / total_segments)

    Higher is better (fewer segments changed → more interpretable).

    Parameters
    ----------
    X_original, X_cf : np.ndarray, shape (C, T)
    n_segments : int
    tol : float
        Tolerance for detecting "no change".

    Returns
    -------
    float in [0, 1]
    """
    T = X_original.shape[-1]
    boundaries = _segment_boundaries(T, n_segments)

    n_modified = sum(
        1 for start, end in boundaries
        if not np.allclose(X_original[:, start:end], X_cf[:, start:end], atol=tol)
    )

    return 1.0 - (n_modified / n_segments)


# Validity
def validity(
    X_cf: np.ndarray,
    model: Callable,
    target_class: int,
) -> bool:
    """Whether the counterfactual achieves the desired class.

    Parameters
    ----------
    X_cf : np.ndarray, shape (C, T)
    model : callable
        (N, C, T) → (N,) class predictions or probabilities.
    target_class : int

    Returns
    -------
    bool
    """
    out = np.asarray(model(X_cf[np.newaxis])).ravel()
    if out.size == 1:
        pred = int(out[0] > 0.5)
    else:
        pred = int(np.argmax(out))
    return pred == target_class


# Diversity
def diversity(
    counterfactuals: np.ndarray,
    dtw_window: float = 0.1,
) -> float:
    """Average pairwise DTW distance across a set of counterfactuals.

    Diversity = (2 / n(n-1)) * ΣΣ DTW(X'_i, X'_j)  for i < j

    Parameters
    ----------
    counterfactuals : np.ndarray, shape (N, C, T)
    dtw_window : float

    Returns
    -------
    float
    """
    N = len(counterfactuals)
    if N < 2:
        return 0.0

    T = counterfactuals.shape[-1]
    window = max(1, int(dtw_window * T))
    total = 0.0
    count = 0

    for i in range(N):
        for j in range(i + 1, N):
            total += _multivariate_dtw(counterfactuals[i], counterfactuals[j], window=window)
            count += 1

    return total / count if count > 0 else 0.0


# Batch evaluation summary
def evaluate_batch(results: list, dtw_window: float = 0.1) -> dict:
    """Compute aggregate evaluation metrics over a list of CounterfactualResults.

    Parameters
    ----------
    results : list of CounterfactualResult
    dtw_window : float

    Returns
    -------
    dict with keys: validity_rate, proximity_mean, proximity_std,
                    sparsity_mean, sparsity_std, diversity
    """
    valid_flags = [r.is_valid for r in results]
    proxs = [r.proximity_score for r in results]
    spars = [r.sparsity_score for r in results]
    cfs = np.stack([r.counterfactual for r in results])

    return {
        "n_instances":    len(results),
        "validity_rate":  float(np.mean(valid_flags)),
        "proximity_mean": float(np.mean(proxs)),
        "proximity_std":  float(np.std(proxs)),
        "sparsity_mean":  float(np.mean(spars)),
        "sparsity_std":   float(np.std(spars)),
        "diversity":      diversity(cfs, dtw_window=dtw_window),
    }
