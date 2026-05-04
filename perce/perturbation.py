"""
Hierarchical segment-channel perturbation algorithm.

Implements Algorithm 1 from the PerCE paper (Bayrak & Bach, IEEE Access 2025):

1. Sort channels by importance (descending).
2. For each channel (most → least important):
   a. Sort segments by importance (descending).
   b. For each segment: interpolate X[c,s] toward candidate[c,s].
   c. If model now predicts target class → record and optionally return.
3. If still not valid after all segment attempts: do full-channel
   interpolation (channel-wise refinement, controlled by beta).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .importance import _segment_boundaries


def hierarchical_perturb(
    X_query: np.ndarray,
    candidate: np.ndarray,
    model: Callable,
    ch_importance: np.ndarray,
    seg_importance: np.ndarray,
    n_segments: int,
    target_class: int,
    alpha: float = 0.5,
    beta: float = 0.6,
) -> tuple:
    """Run the hierarchical perturbation strategy.

    Parameters
    ----------
    X_query : np.ndarray, shape (C, T)
    candidate : np.ndarray, shape (C, T)
        InSample anchor from target class.
    model : callable
        Classifier: (N, C, T) → (N,) predictions.
    ch_importance : np.ndarray, shape (C,)
    seg_importance : np.ndarray, shape (n_segments,)
    n_segments : int
    target_class : int
    alpha : float
        Interpolation weight for segment-level step.
    beta : float
        Interpolation weight for full-channel fallback step.

    Returns
    -------
    cf : np.ndarray, shape (C, T)
        The best counterfactual found.
    channels_modified : list[int]
        Channel indices that were modified.
    segments_modified : list[tuple[int, int]]
        (channel, segment_index) pairs that were modified.
    """
    C, T = X_query.shape
    boundaries = _segment_boundaries(T, n_segments)

    # Working copy
    cf = X_query.copy()
    channels_modified = []
    segments_modified = []

    # Sort channels by importance
    ch_order = np.argsort(ch_importance)[::-1]    # highest first
    seg_order = np.argsort(seg_importance)[::-1]  # highest first (global)

    # Phase 1: Segment-wise perturbation
    for c in ch_order:
        for s in seg_order:
            start, end = boundaries[s]

            # Try interpolation at this (channel, segment)
            cf_trial = cf.copy()
            cf_trial[c, start:end] = (
                (1 - alpha) * cf[c, start:end]
                + alpha * candidate[c, start:end]
            )

            pred = _predict_single(model, cf_trial)

            if pred == target_class:
                # Accept the change
                cf = cf_trial
                if c not in channels_modified:
                    channels_modified.append(int(c))
                segments_modified.append((int(c), int(s)))
                return cf, channels_modified, segments_modified
            else:
                # Check if we're improving (moving closer to target)
                # Accept if it moves us in the right direction
                prev_conf = _target_confidence(model, cf, target_class)
                new_conf = _target_confidence(model, cf_trial, target_class)

                if new_conf > prev_conf:
                    cf = cf_trial
                    if c not in channels_modified:
                        channels_modified.append(int(c))
                    segments_modified.append((int(c), int(s)))

        # After trying all segments for this channel, check again
        if _predict_single(model, cf) == target_class:
            return cf, channels_modified, segments_modified

    # Phase 2: Channel-wise refinement (fallback)
    # If segment-level wasn't enough, try full-channel interpolation
    for c in ch_order:
        cf_trial = cf.copy()
        cf_trial[c, :] = (
            (1 - beta) * cf[c, :]
            + beta * candidate[c, :]
        )

        pred = _predict_single(model, cf_trial)
        cf = cf_trial
        if c not in channels_modified:
            channels_modified.append(int(c))

        if pred == target_class:
            return cf, channels_modified, segments_modified

    # Return best effort even if still not valid
    return cf, channels_modified, segments_modified


# Helpers
def _predict_single(model: Callable, X: np.ndarray) -> int:
    """Predict class for a single (C, T) instance."""
    out = np.asarray(model(X[np.newaxis])).ravel()
    if out.size == 1:
        return int(out[0] > 0.5)
    return int(np.argmax(out))


def _target_confidence(model: Callable, X: np.ndarray, target: int) -> float:
    """Return model's confidence for the target class on a (C, T) instance."""
    out = np.asarray(model(X[np.newaxis]), dtype=float).ravel()
    if out.size == 1:
        # Binary: target=1 → confidence = out[0]; target=0 → 1-out[0]
        return float(out[0]) if target == 1 else float(1.0 - out[0])
    if target < len(out):
        return float(out[target])
    return float(np.max(out))
