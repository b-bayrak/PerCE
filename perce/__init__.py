"""
PerCE: Hierarchical Perturbation-Based Counterfactual Explanations
for Multivariate Time Series Classification.

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
__author__ = "Betül Bayrak"
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
