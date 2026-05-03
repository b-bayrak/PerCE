"""
PerCE Quick Start — 30-second demo with synthetic data.

Run:  python examples/quick_start.py

No downloads required. Shows the full pipeline:
  fit → explain → evaluate → print summary.
"""

import numpy as np
from perce import PerCEExplainer, evaluate_batch

# ── Synthetic data ────────────────────────────────────────────────────
# 6 channels, 300 time points, binary labels based on mean of channel 0
rng = np.random.default_rng(0)
N, C, T = 80, 6, 300

X_train = rng.normal(0, 1, (N, C, T)).astype(np.float32)
y_train = (X_train[:, 0, 50:150].mean(axis=1) > 0).astype(int)

X_test  = rng.normal(0, 1, (20, C, T)).astype(np.float32)
y_test  = (X_test[:, 0, 50:150].mean(axis=1) > 0).astype(int)

# ── Simple model (swap for your real one) ────────────────────────────
def my_model(X_np: np.ndarray) -> np.ndarray:
    """Model callable: (N, C, T) → (N, n_classes) probabilities."""
    means = X_np[:, 0, 50:150].mean(axis=1)
    p1 = 1.0 / (1.0 + np.exp(-2 * means))   # sigmoid
    return np.stack([1 - p1, p1], axis=1)

# ── Fit ───────────────────────────────────────────────────────────────
print("Fitting PerCEExplainer...")
explainer = PerCEExplainer(
    model=my_model,
    n_segments=6,   # divide each series into 6 segments
    alpha=0.5,      # segment interpolation strength
    beta=0.6,       # channel fallback strength
    k=3,            # k-nearest neighbours
    random_state=42,
)
explainer.fit(X_train, y_train)
print(explainer)

# ── Explain a single instance ─────────────────────────────────────────
query        = X_test[0]
target_class = 1 - int(y_test[0])   # flip the predicted class

print(f"\nGenerating counterfactual for target class {target_class}...")
result = explainer.explain(query, target_class=target_class)

print("\n" + result.summary())
print(f"\nChannels modified : {result.channels_modified}")
print(f"Segments modified : {len(result.segments_modified)}")

# ── Batch explanation + evaluation ────────────────────────────────────
print("\nRunning batch explanation on 20 test instances...")
results = explainer.explain_batch(
    X_test,
    target_classes=[1 - int(y) for y in y_test],
    verbose=False,
)

summary = evaluate_batch(results)
print("\n── Batch Evaluation ─────────────────────────────────────────")
print(f"  Validity rate : {summary['validity_rate']:.2%}")
print(f"  Proximity     : {summary['proximity_mean']:.4f} ± {summary['proximity_std']:.4f}")
print(f"  Sparsity      : {summary['sparsity_mean']:.4f} ± {summary['sparsity_std']:.4f}")
print(f"  Diversity     : {summary['diversity']:.4f}")
print("─" * 55)
print("\nDone! For ECG demo on real data, see: notebooks/01_ECG_demo.ipynb")
