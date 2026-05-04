# PerCE: Hierarchical Perturbation-Based Counterfactual Explanations

[![PyPI version](https://img.shields.io/pypi/v/perce.svg)](https://pypi.org/project/perce/)
[![Python 3.9+](https://img.shields.io/pypi/pyversions/perce.svg)](https://pypi.org/project/perce/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/b-bayrak/PerCE/actions/workflows/ci.yml/badge.svg)](https://github.com/b-bayrak/PerCE/actions)
[![IEEE Access](https://img.shields.io/badge/IEEE%20Access-10.1109%2FACCESS.2025.3639125-blue)](https://doi.org/10.1109/ACCESS.2025.3639125)

**PerCE** generates plausible counterfactual explanations for multivariate time series classification.

> *"What minimal change to this ECG would cause the model to predict a different diagnosis?"*

PerCE answers that question using a **hierarchical perturbation strategy** guided by permutation-based feature importance, operating at both temporal-segment and channel levels. It anchors every counterfactual to a real training instance (*InSample*), enabling physiologically plausible results.

---

## Paper

> **Bayrak, B. & Bach, K. (2025).** PerCE: Hierarchical Perturbation-Based Counterfactual Explanations for Multivariate Time Series Classification. *IEEE Access*. [DOI: 10.1109/ACCESS.2025.3639125](https://doi.org/10.1109/ACCESS.2025.3639125)

If you use PerCE in your work, please cite:

```bibtex
@article{bayrak2025perce,
  title   = {{PerCE}: Hierarchical Perturbation-Based Counterfactual Explanations
             for Multivariate Time Series Classification},
  author  = {Bayrak, Bet\"{u}l and Bach, Kerstin},
  journal = {IEEE Access},
  year    = {2025},
  doi     = {10.1109/ACCESS.2025.3639125}
}
```

---

## Key Results

Evaluated on the open [CODE-test ECG dataset](https://zenodo.org/records/3765780) (827 12-lead recordings, cardiologist-annotated), PerCE substantially outperforms the InSample baseline:

```python
from perce import evaluate_batch

summary = evaluate_batch(results)
print(f"Validity:  {summary['validity_rate']:.2%}")
print(f"Proximity: {summary['proximity_mean']:.3f} ± {summary['proximity_std']:.3f}")
print(f"Sparsity:  {summary['sparsity_mean']:.3f} ± {summary['sparsity_std']:.3f}")
print(f"Diversity: {summary['diversity']:.3f}")
```

| Metric | InSample baseline | **PerCE** | Improvement |
|--------|:-----------------:|:---------:|:-----------:|
| Validity ↑ | 0.65 ± 0.35 | **0.98 ± 0.05** | +51% |
| Proximity ↓ | 200 ± 150 | **50 ± 25** | −75% |
| Sparsity ↑ | 0.70 ± 0.15 | **0.40 ± 0.12** | +43% |
| Diversity | 0.30 ± 0.10 | 0.20 ± 0.08 | (tunable) |

*Validity: 1.0 = always achieves class change. Proximity: lower = closer to original. Sparsity: lower = fewer segments modified.*

---

## Installation

```bash
pip install perce
```

With ECG demo dependencies (h5py, matplotlib):

```bash
pip install "perce[ecg]"
```

**Requirements:** Python ≥ 3.9, NumPy ≥ 1.24, SciPy ≥ 1.10, scikit-learn ≥ 1.3. No Java. No external REST API.

---

## Quick Start

```python
import numpy as np
from perce import PerCEExplainer

# 1. Wrap your model
#    Must accept (N, C, T) and return (N,) class predictions
def my_model(X):
    # PyTorch example:
    # import torch
    # with torch.no_grad():
    #     return net(torch.tensor(X)).argmax(dim=1).numpy()
    return np.zeros(len(X), dtype=int)   # stub

# 2. Fit (store training data for InSample candidate selection) 
exp = PerCEExplainer(
    model=my_model,
    n_segments=10,   # divide each time series into 10 segments
    alpha=0.5,       # segment-level interpolation strength
    beta=0.6,        # channel-level fallback strength
    k=5,             # k-nearest neighbours for candidate selection
)
exp.fit(X_train, y_train)   # X_train shape: (N, C, T)

# 3. Explain a single instance
result = exp.explain(X_query, target_class=1)   # X_query shape: (C, T)
print(result.summary())

# 4. Access everything 
cf  = result.counterfactual       # shape (C, T) — the explanation
print("Valid?    ", result.is_valid)
print("Proximity:", result.proximity_score)
print("Sparsity: ", result.sparsity_score)
print("Channels modified:", result.channels_modified)
```

### Batch explanation

```python
results = exp.explain_batch(X_test, target_classes=1, verbose=True)

from perce import evaluate_batch
summary = evaluate_batch(results)
print(f"Validity rate: {summary['validity_rate']:.2%}")
print(f"Proximity:     {summary['proximity_mean']:.3f} ± {summary['proximity_std']:.3f}")
```

---

## How It Works

PerCE follows a three-stage hierarchical algorithm:

```
Query X (C channels × T time points)
         │
         ▼
① Feature Importance          ② InSample Candidate
  • Channel-level (Ich)          k-NN from target class
  • Segment-level (Iseg)         via DTW distance
  Both: permutation-based,
  model-agnostic
         │                              │
         └──────────────┬───────────────┘
                        ▼
             ③ Hierarchical Perturbation
               For each channel c (most→least important):
                 For each segment s (most→least important):
                   X'[c,s] = (1−α)·X[c,s] + α·Xcand[c,s]
                   If model predicts target → return X'
               Fallback: full-channel replacement (beta)
                        │
                        ▼
             Counterfactual X' ← real ECG pattern,
             minimal changes, high validity
```
---

**Why hierarchical?**
> Traditional perturbation methods modify all features blindly. PerCE focuses on the most informative channels and time windows first, this facilitates producing sparser, more clinically meaningful explanations.

**Why InSample?** 
> By anchoring to real training instances, every generated counterfactual is guaranteed to be within the data distribution. No out-of-distribution artifacts.

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| [`01_ECG_demo.ipynb`](notebooks/01_ECG_demo.ipynb) | Full pipeline on CODE-test — reproduces Table 1 from paper |
| [`02_custom_model.ipynb`](notebooks/02_custom_model.ipynb) | How to plug in your own PyTorch / sklearn model |
| [`03_evaluation.ipynb`](notebooks/03_evaluation.ipynb) | Comprehensive evaluation |

---

## Related Packages

This package is part of a growing XAI ecosystem from NTNU's NorwAI Centre:

| Package                                                    | What it does | Paper |
|------------------------------------------------------------|-------------|-------|
| **[PerCE](https://pypi.org/project/perce/)**               | Counterfactual explanations for **time series** | [IEEE Access 2025](https://doi.org/10.1109/ACCESS.2025.3639125) |
| **[PertCF](https://github.com/b-bayrak/PertCF-Explainer)** | Counterfactual explanations for **tabular data** | [SGAI 2023](https://doi.org/10.1007/978-3-031-47994-6_13) |
| **[CEval](https://pypi.org/project/CEval/)**               | Evaluation framework for **any** counterfactual method | [IEEE Access 2024](https://doi.org/10.1109/ACCESS.2024.3466475) |

---

## Repository Structure

```
perce/
├── perce/
│   ├── __init__.py          # Public API
│   ├── explainer.py         # PerCEExplainer + CounterfactualResult
│   ├── importance.py        # Channel- and segment-level permutation importance
│   ├── neighbors.py         # DTW-based k-NN candidate selection
│   ├── perturbation.py      # Hierarchical interpolation (Algorithm 1)
│   └── metrics.py           # Proximity, Sparsity, Validity, Diversity
├── notebooks/
│   ├── 01_ECG_demo.ipynb
│   ├── 02_custom_model.ipynb
│   └── 03_evaluation.ipynb
├── tests/
│   └── test_perce.py
├── pyproject.toml
└── README.md
```

---

## API Reference

### `PerCEExplainer`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | — | Callable: `(N, C, T) → (N,)` class predictions |
| `n_segments` | `10` | Number of temporal segments |
| `alpha` | `0.5` | Segment-level interpolation weight |
| `beta` | `0.6` | Channel-level fallback interpolation weight |
| `k` | `5` | Nearest neighbours for candidate selection |
| `dtw_window` | `0.1` | Sakoe-Chiba band (fraction of T) |
| `random_state` | `42` | Reproducibility seed |

### `CounterfactualResult`

| Attribute | Type | Description |
|-----------|------|-------------|
| `.counterfactual` | `ndarray (C,T)` | The generated explanation |
| `.is_valid` | `bool` | Did it achieve the target class? |
| `.proximity_score` | `float` | Normalised DTW distance (lower=better) |
| `.sparsity_score` | `float` | Fraction of segments unchanged (higher=better) |
| `.channels_modified` | `list[int]` | Which channels were touched |
| `.candidate` | `ndarray (C,T)` | InSample anchor used |

---

## Running Tests

```bash
git clone https://github.com/b-bayrak/PerCE
cd PerCE
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Acknowledgements

This work was supported by the Research Council of Norway through the **SFI NorwAI** (Norwegian Research Center for AI Innovation), grant number 309834.

---

## License

MIT © Betül Bayrak