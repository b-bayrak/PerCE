"""
Permutation-based feature importance for multivariate time series.

Implements both channel-level and segment-level importance as described
in Section III-B of the PerCE paper (Bayrak & Bach, IEEE Access 2025).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


# Channel-level importance
def channel_importance(
    model: Callable,
    X: np.ndarray,
    n_samples: int = 20,
    rng: Optional[np.random.Generator] = None,
    baseline_value: float = 0.0,
) -> np.ndarray:
    """Compute permutation-based importance for each channel.

    For each channel c, the importance I_ch(c) is the drop in model
    confidence (for the predicted class) when channel c is replaced by
    a baseline value:

        I_ch(c) = score(f, X) - score(f, X^(c))

    where X^(c) is X with channel c replaced by `baseline_value`.

    Parameters
    ----------
    model : callable
        Accepts (N, C, T) → returns (N,) class probabilities or logits.
    X : np.ndarray, shape (C, T)
        A single multivariate time series instance.
    n_samples : int
        Unused — kept for API symmetry. Permutation here is deterministic.
    baseline_value : float
        Value used to replace a channel (default: 0.0, i.e. zeroing).

    Returns
    -------
    importance : np.ndarray, shape (C,)
        Importance score for each channel. Higher = more important.
    """
    rng = rng or np.random.default_rng(42)
    C, T = X.shape

    # Baseline: model confidence on original instance
    base_score = _model_score(model, X)

    importance = np.zeros(C)
    for c in range(C):
        X_perturbed = X.copy()
        X_perturbed[c, :] = baseline_value
        perturbed_score = _model_score(model, X_perturbed)
        importance[c] = base_score - perturbed_score

    # Clip negatives to 0 (negative drop = channel hurts performance)
    return importance


# Segment-level importance
def segment_importance(
    model: Callable,
    X: np.ndarray,
    n_segments: int = 10,
    n_samples: int = 20,
    rng: Optional[np.random.Generator] = None,
    baseline_value: float = 0.0,
) -> np.ndarray:
    """Compute permutation-based importance for each temporal segment.

    For each segment s, the importance I_seg(s) is the drop in model
    confidence when ALL channels within segment s are replaced:

        I_seg(s) = score(f, X) - score(f, X^[s_i:s_{i+1}])

    Parameters
    ----------
    model : callable
        Accepts (N, C, T) → returns (N,) class probabilities or logits.
    X : np.ndarray, shape (C, T)
        A single multivariate time series instance.
    n_segments : int
        Number of non-overlapping equal-length segments.
    n_samples : int
        Unused — kept for API symmetry.
    baseline_value : float
        Value used to replace a segment (default: 0.0).

    Returns
    -------
    importance : np.ndarray, shape (n_segments,)
        Importance score for each segment. Higher = more important.
    """
    rng = rng or np.random.default_rng(42)
    C, T = X.shape
    seg_len = T // n_segments
    boundaries = _segment_boundaries(T, n_segments)

    base_score = _model_score(model, X)

    importance = np.zeros(n_segments)
    for s, (start, end) in enumerate(boundaries):
        X_perturbed = X.copy()
        X_perturbed[:, start:end] = baseline_value   # all channels, this segment
        perturbed_score = _model_score(model, X_perturbed)
        importance[s] = base_score - perturbed_score

    return importance


def _model_score(model: Callable, X: np.ndarray) -> float:
    """Return the model's confidence on a single (C, T) instance.

    Handles:
    - Models returning (N, K) probability arrays → max probability
    - Models returning (N,) scalar predictions → direct value
    - Binary sigmoid output (scalar in [0,1])
    """
    X_batch = X[np.newaxis]   # (1, C, T)
    out = np.asarray(model(X_batch), dtype=float).ravel()

    if out.size == 1:
        # Binary sigmoid output or scalar
        p = float(out[0])
        return max(p, 1.0 - p)   # confidence = distance from 0.5
    else:
        # Softmax output (K classes)
        return float(np.max(out))


def _segment_boundaries(T: int, n_segments: int) -> list:
    """Return list of (start, end) index pairs for uniform segmentation."""
    seg_len = T // n_segments
    boundaries = []
    for s in range(n_segments):
        start = s * seg_len
        end = start + seg_len if s < n_segments - 1 else T
        boundaries.append((start, end))
    return boundaries
