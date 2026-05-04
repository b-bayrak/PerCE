"""
InSample candidate selection via DTW-based k-nearest neighbours.

Implements Step I of the PerCE algorithm (Section III-A):
  Find the k nearest neighbours from the target class using DTW,
  then select the candidate with minimum DTW distance to the query.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def find_candidate(
    X_query: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    target_class: int,
    k: int = 5,
    dtw_window: float = 0.1,
) -> np.ndarray:
    """Find the best InSample candidate from the target class.

    Parameters
    ----------
    X_query : np.ndarray, shape (C, T)
        The query instance.
    X_train : np.ndarray, shape (N, C, T)
        All training instances.
    y_train : np.ndarray, shape (N,)
        Class labels for training instances.
    target_class : int
        The class we want the counterfactual to belong to.
    k : int
        Number of nearest neighbours to retrieve.
    dtw_window : float
        Sakoe-Chiba band as fraction of T.

    Returns
    -------
    candidate : np.ndarray, shape (C, T)
        The nearest training instance from the target class.
    """
    # Filter to target class
    mask = (np.asarray(y_train) == target_class)
    X_target = X_train[mask]

    if len(X_target) == 0:
        raise ValueError(
            f"No training instances found for target class {target_class}. "
            "Check your y_train labels."
        )

    # Compute DTW distances from query to all target-class instances
    T = X_query.shape[-1]
    window = max(1, int(dtw_window * T))

    distances = np.array([
        _multivariate_dtw(X_query, X_target[i], window=window)
        for i in range(len(X_target))
    ])

    # Select k nearest, then return the single closest
    k = min(k, len(X_target))
    knn_indices = np.argsort(distances)[:k]
    best_idx = knn_indices[0]   # already the global minimum

    return X_target[best_idx].copy()


# DTW implementation
def _multivariate_dtw(
    X: np.ndarray,
    Y: np.ndarray,
    window: Optional[int] = None,
) -> float:
    """Compute multivariate DTW distance between two (C, T) instances.

    Uses independent DTW: sum of per-channel DTW distances.
    Optionally uses the Sakoe-Chiba band constraint.

    Parameters
    ----------
    X, Y : np.ndarray, shape (C, T)
    window : int or None
        Sakoe-Chiba band half-width. None = unconstrained.

    Returns
    -------
    float : total DTW distance (sum across channels)
    """
    # Pad to same length if needed
    C = X.shape[0]
    total = 0.0
    for c in range(C):
        total += _dtw_1d(X[c], Y[c], window=window)
    return total


def _dtw_1d(x: np.ndarray, y: np.ndarray, window: Optional[int] = None) -> float:
    """Standard DTW for two 1-D sequences with optional Sakoe-Chiba band."""
    n, m = len(x), len(y)

    # Cost matrix
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = 1 if window is None else max(1, i - window)
        j_end = m if window is None else min(m, i + window)
        for j in range(j_start, j_end + 1):
            d = (x[i - 1] - y[j - 1]) ** 2
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    return float(np.sqrt(cost[n, m]))


def dtw_distance(X: np.ndarray, Y: np.ndarray, window: float = 0.1) -> float:
    """Public API: DTW distance between two (C, T) arrays.

    Parameters
    ----------
    X, Y : np.ndarray, shape (C, T)
    window : float
        Sakoe-Chiba band as fraction of T.

    Returns
    -------
    float
    """
    T = X.shape[-1]
    w = max(1, int(window * T))
    return _multivariate_dtw(X, Y, window=w)
