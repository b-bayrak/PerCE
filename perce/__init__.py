"""
PerCE: Hierarchical Perturbation-Based Counterfactual Explanations
for Multivariate Time Series Classification.

Quick start
-----------
>>> import numpy as np
>>> from perce import PerCEExplainer
>>>
>>> # Your model: callable (N, C, T) → (N,) class indices
>>> exp = PerCEExplainer(model=my_model, n_segments=10)
>>> exp.fit(X_train, y_train)
>>>
>>> result = exp.explain(X_query, target_class=1)
>>> print(result.summary())

Reference
---------
Bayrak, B. & Bach, K. (2025). PerCE: Hierarchical Perturbation-Based
Counterfactual Explanations for Multivariate Time Series Classification.
IEEE Access. DOI: 10.1109/ACCESS.2025.3639125
"""

from .explainer import PerCEExplainer, CounterfactualResult
from .metrics import proximity, sparsity, validity, diversity, evaluate_batch
from .neighbors import dtw_distance
from .importance import channel_importance, segment_importance

__version__ = "0.1.0"
__author__ = "Betül Bayrak, Kerstin Bach"
__license__ = "MIT"

__all__ = [
    # Core API
    "PerCEExplainer",
    "CounterfactualResult",
    # Metrics
    "proximity",
    "sparsity",
    "validity",
    "diversity",
    "evaluate_batch",
    # Utilities
    "dtw_distance",
    "channel_importance",
    "segment_importance",
]
