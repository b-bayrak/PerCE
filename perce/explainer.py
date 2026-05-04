"""
PerCE: Hierarchical Perturbation-Based Counterfactual Explanations
for Multivariate Time Series Classification.

Reference:
    Bayrak, B. & Bach, K. (2025). PerCE: Hierarchical Perturbation-Based
    Counterfactual Explanations for Multivariate Time Series Classification.
    IEEE Access. DOI: 10.1109/ACCESS.2025.3639125
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .importance import channel_importance, segment_importance
from .metrics import diversity, proximity, sparsity, validity
from .neighbors import find_candidate
from .perturbation import hierarchical_perturb


@dataclass
class CounterfactualResult:
    """Container for a single counterfactual explanation.

    Attributes
    ----------
    counterfactual : np.ndarray
        The generated counterfactual of shape (C, T).
    original : np.ndarray
        The original query instance of shape (C, T).
    target_class : int
        The target class the counterfactual achieves.
    predicted_class : int
        Class predicted by the model for the counterfactual.
    is_valid : bool
        Whether the counterfactual successfully changed the class.
    channels_modified : list[int]
        Indices of channels that were modified.
    segments_modified : list[tuple[int, int]]
        (channel, segment) pairs that were modified.
    proximity_score : float
        DTW distance between original and counterfactual, normalised by C*T.
    sparsity_score : float
        Fraction of segments *not* modified (higher = sparser).
    candidate : np.ndarray
        The InSample candidate used as anchor.
    """

    counterfactual: np.ndarray
    original: np.ndarray
    target_class: int
    predicted_class: int
    is_valid: bool
    channels_modified: list
    segments_modified: list
    proximity_score: float
    sparsity_score: float
    candidate: np.ndarray
    metadata: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "PerCE Counterfactual Result",
            "=" * 40,
            f"  Valid          : {self.is_valid}",
            f"  Target class   : {self.target_class}",
            f"  Predicted class: {self.predicted_class}",
            f"  Proximity      : {self.proximity_score:.4f}",
            f"  Sparsity       : {self.sparsity_score:.4f}",
            f"  Channels mod.  : {self.channels_modified}",
            f"  Segments mod.  : {len(self.segments_modified)}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"CounterfactualResult(valid={self.is_valid}, "
            f"proximity={self.proximity_score:.3f}, "
            f"sparsity={self.sparsity_score:.3f})"
        )


class PerCEExplainer:
    """Hierarchical Perturbation-Based Counterfactual Explainer for
    multivariate time series classification.

    PerCE generates plausible counterfactual explanations by:
    1. Computing permutation-based feature importance at channel and
       segment levels.
    2. Selecting an InSample candidate from the target class using
       DTW-based k-nearest neighbours.
    3. Applying hierarchical interpolation: segment-level perturbations
       in important channels first, escalating to full-channel replacement.

    Parameters
    ----------
    model : callable
        A trained classifier. Must accept arrays of shape (N, C, T) and
        return predicted class indices (or probabilities) as a 1-D array
        of length N. Compatible with PyTorch, TensorFlow/Keras, and
        scikit-learn pipelines via a thin wrapper.
    n_segments : int, default=10
        Number of non-overlapping temporal segments.
    alpha : float, default=0.5
        Interpolation weight for segment-level perturbation.
        X'[c,s] = (1-alpha)*X[c,s] + alpha*Xcand[c,s]
    beta : float, default=0.6
        Interpolation weight for full-channel replacement.
        X'[c,:] = (1-beta)*X[c,:] + beta*Xcand[c,:]
    k : int, default=5
        Number of nearest neighbours to consider for candidate selection.
    dtw_window : float, default=0.1
        DTW Sakoe-Chiba band as fraction of sequence length.
    n_importance_samples : int, default=20
        Number of permutation samples for importance estimation.
    random_state : int or None, default=42
        Random seed for reproducibility.

    """

    def __init__(
        self,
        model: Callable,
        n_segments: int = 10,
        alpha: float = 0.5,
        beta: float = 0.6,
        k: int = 5,
        dtw_window: float = 0.1,
        n_importance_samples: int = 20,
        random_state: Optional[int] = 42,
    ):
        self.model = model
        self.n_segments = n_segments
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.dtw_window = dtw_window
        self.n_importance_samples = n_importance_samples
        self.random_state = random_state

        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._is_fitted: bool = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "PerCEExplainer":
        """Store training data for InSample candidate selection.

        Parameters
        ----------
        X_train : np.ndarray, shape (N, C, T)
            Training instances.
        y_train : np.ndarray, shape (N,)
            Class labels.

        Returns
        -------
        self
        """
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train)

        if X_train.ndim != 3:
            raise ValueError(
                f"X_train must be 3-D (N, C, T), got shape {X_train.shape}"
            )
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must have the same length.")

        self._X_train = X_train
        self._y_train = y_train
        self._is_fitted = True
        return self

    def explain(
        self,
        X_query: np.ndarray,
        target_class: int,
    ) -> CounterfactualResult:
        """Generate a counterfactual explanation for a single instance.

        Parameters
        ----------
        X_query : np.ndarray, shape (C, T)
            The query instance to explain.
        target_class : int
            The desired output class for the counterfactual.

        Returns
        -------
        CounterfactualResult
        """
        self._check_is_fitted()
        X_query = np.asarray(X_query, dtype=float)

        if X_query.ndim != 2:
            raise ValueError(
                f"X_query must be 2-D (C, T), got shape {X_query.shape}"
            )

        rng = np.random.default_rng(self.random_state)
        C, T = X_query.shape

        # Step 1: Feature importance
        ch_imp = channel_importance(
            model=self.model,
            X=X_query,
            n_samples=self.n_importance_samples,
            rng=rng,
        )
        seg_imp = segment_importance(
            model=self.model,
            X=X_query,
            n_segments=self.n_segments,
            n_samples=self.n_importance_samples,
            rng=rng,
        )

        # Step 2: InSample candidate selection
        candidate = find_candidate(
            X_query=X_query,
            X_train=self._X_train,
            y_train=self._y_train,
            target_class=target_class,
            k=self.k,
            dtw_window=self.dtw_window,
        )

        # Step 3: Hierarchical perturbation
        cf, channels_modified, segments_modified = hierarchical_perturb(
            X_query=X_query,
            candidate=candidate,
            model=self.model,
            ch_importance=ch_imp,
            seg_importance=seg_imp,
            n_segments=self.n_segments,
            target_class=target_class,
            alpha=self.alpha,
            beta=self.beta,
        )

        # Step 4: Evaluate
        pred = int(self._predict_single(cf))
        is_val = validity(cf, self.model, target_class)
        prox = proximity(X_query, cf, dtw_window=self.dtw_window)
        spar = sparsity(X_query, cf, n_segments=self.n_segments)

        return CounterfactualResult(
            counterfactual=cf,
            original=X_query,
            target_class=target_class,
            predicted_class=pred,
            is_valid=is_val,
            channels_modified=channels_modified,
            segments_modified=segments_modified,
            proximity_score=prox,
            sparsity_score=spar,
            candidate=candidate,
            metadata={
                "channel_importance": ch_imp,
                "segment_importance": seg_imp,
                "alpha": self.alpha,
                "beta": self.beta,
                "n_segments": self.n_segments,
            },
        )

    # ------------------------------------------------------------------
    # Explain (batch)
    # ------------------------------------------------------------------

    def explain_batch(
        self,
        X_queries: np.ndarray,
        target_classes,
        verbose: bool = True,
    ) -> list:
        """Generate counterfactuals for multiple query instances.

        Parameters
        ----------
        X_queries : np.ndarray, shape (N, C, T)
        target_classes : int or array-like of int
            Single class (applied to all) or one per instance.
        verbose : bool
            Print progress.

        Returns
        -------
        list of CounterfactualResult
        """
        X_queries = np.asarray(X_queries, dtype=float)
        N = len(X_queries)

        if isinstance(target_classes, int):
            target_classes = [target_classes] * N
        else:
            target_classes = list(target_classes)
            if len(target_classes) != N:
                raise ValueError(
                    "target_classes must be a scalar or have length N."
                )

        results = []
        for i, (X, tc) in enumerate(zip(X_queries, target_classes)):
            if verbose and (i % 10 == 0 or i == N - 1):
                print(f"  Explaining instance {i+1}/{N}...")
            results.append(self.explain(X, tc))
        return results
    @staticmethod
    def diversity_score(results: list, dtw_window: float = 0.1) -> float:
        """Compute average pairwise DTW diversity across a list of results."""
        cfs = np.stack([r.counterfactual for r in results])
        return diversity(cfs, dtw_window=dtw_window)

    def _predict_single(self, X: np.ndarray) -> int:
        """Predict class for a single (C, T) instance."""
        X_batch = X[np.newaxis]          # (1, C, T)
        pred = self.model(X_batch)
        pred = np.asarray(pred).ravel()
        # Handle probability output (N, K) vs class index output (N,)
        if pred.ndim == 0:
            return int(pred)
        if pred.shape[0] > 1 and pred[0] <= 1.0:
            # Looks like probabilities for binary case
            return int(pred[0] > 0.5)
        return int(np.argmax(pred)) if pred.ndim > 1 else int(pred[0])

    def _check_is_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(
                "PerCEExplainer must be fitted before calling explain(). "
                "Call explainer.fit(X_train, y_train) first."
            )

    def __repr__(self) -> str:
        return (
            f"PerCEExplainer("
            f"n_segments={self.n_segments}, "
            f"alpha={self.alpha}, "
            f"beta={self.beta}, "
            f"k={self.k})"
        )
