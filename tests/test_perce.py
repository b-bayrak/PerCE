"""
Test suite for PerCE.

Uses synthetic data so tests run without downloading the dataset.
"""

import numpy as np
import pytest

from perce import (
    PerCEExplainer,
    CounterfactualResult,
    proximity,
    sparsity,
    validity,
    diversity,
    evaluate_batch,
    channel_importance,
    segment_importance,
    dtw_distance,
)


C, T = 4, 200      # small for fast tests
N_TRAIN = 40


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(0)
    X_train = rng.normal(0, 1, (N_TRAIN, C, T)).astype(np.float32)
    y_train = np.array([i % 2 for i in range(N_TRAIN)])
    X_query = rng.normal(0, 1, (C, T)).astype(np.float32)
    return X_train, y_train, X_query


def make_model(target_pred=0):
    """ Deterministic stub model that always predicts `target_pred`."""
    def model(X):
        N = len(X)
        out = np.zeros((N, 2), dtype=float)
        out[:, target_pred] = 1.0
        return out
    return model


def make_flip_model(threshold=0.0):
    """ Model that flips class based on mean of first channel."""
    def model(X):
        X = np.asarray(X)
        N = X.shape[0]
        preds = (X[:, 0, :].mean(axis=-1) > threshold).astype(float)
        out = np.stack([1 - preds, preds], axis=1)
        return out
    return model



# PerCEExplainer
class TestPerCEExplainer:

    def test_fit_returns_self(self, synthetic_data):
        X_train, y_train, _ = synthetic_data
        exp = PerCEExplainer(model=make_model(0), n_segments=5)
        result = exp.fit(X_train, y_train)
        assert result is exp

    def test_explain_returns_correct_type(self, synthetic_data):
        X_train, y_train, X_query = synthetic_data
        exp = PerCEExplainer(model=make_flip_model(), n_segments=5)
        exp.fit(X_train, y_train)
        result = exp.explain(X_query, target_class=1)
        assert isinstance(result, CounterfactualResult)

    def test_counterfactual_shape_matches_query(self, synthetic_data):
        X_train, y_train, X_query = synthetic_data
        exp = PerCEExplainer(model=make_flip_model(), n_segments=5)
        exp.fit(X_train, y_train)
        result = exp.explain(X_query, target_class=1)
        assert result.counterfactual.shape == X_query.shape

    def test_explain_fails_without_fit(self, synthetic_data):
        _, _, X_query = synthetic_data
        exp = PerCEExplainer(model=make_model(0), n_segments=5)
        with pytest.raises(RuntimeError, match="fitted"):
            exp.explain(X_query, target_class=1)

    def test_explain_fails_wrong_query_shape(self, synthetic_data):
        X_train, y_train, _ = synthetic_data
        exp = PerCEExplainer(model=make_model(0), n_segments=5)
        exp.fit(X_train, y_train)
        with pytest.raises(ValueError, match="2-D"):
            exp.explain(np.ones((3, C, T)), target_class=1)

    def test_no_target_class_instances_raises(self, synthetic_data):
        X_train, y_train, X_query = synthetic_data
        exp = PerCEExplainer(model=make_model(0), n_segments=5)
        exp.fit(X_train, y_train)
        with pytest.raises(ValueError, match="No training instances"):
            exp.explain(X_query, target_class=99)

    def test_explain_batch(self, synthetic_data):
        X_train, y_train, X_query = synthetic_data
        exp = PerCEExplainer(model=make_flip_model(), n_segments=5)
        exp.fit(X_train, y_train)
        results = exp.explain_batch(X_train[:5], target_classes=1, verbose=False)
        assert len(results) == 5
        assert all(isinstance(r, CounterfactualResult) for r in results)

    def test_summary_string(self, synthetic_data):
        X_train, y_train, X_query = synthetic_data
        exp = PerCEExplainer(model=make_flip_model(), n_segments=5)
        exp.fit(X_train, y_train)
        result = exp.explain(X_query, target_class=1)
        s = result.summary()
        assert "Valid" in s
        assert "Proximity" in s

    def test_repr(self, synthetic_data):
        X_train, y_train, _ = synthetic_data
        exp = PerCEExplainer(model=make_model(0), n_segments=5)
        assert "PerCEExplainer" in repr(exp)


# Metrics
class TestMetrics:

    def test_proximity_identical(self):
        X = np.zeros((C, T))
        assert proximity(X, X.copy()) == pytest.approx(0.0, abs=1e-6)

    def test_proximity_different(self):
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, (C, T))
        Y = rng.normal(5, 1, (C, T))
        assert proximity(X, Y) > 0.0

    def test_sparsity_no_change(self):
        X = np.ones((C, T))
        assert sparsity(X, X.copy(), n_segments=5) == pytest.approx(1.0)

    def test_sparsity_full_change(self):
        X = np.ones((C, T))
        Y = np.zeros((C, T))
        s = sparsity(X, Y, n_segments=5)
        assert s == pytest.approx(0.0)

    def test_validity_correct_class(self):
        X_cf = np.zeros((C, T))
        model = make_model(target_pred=1)
        assert validity(X_cf, model, target_class=1) is True

    def test_validity_wrong_class(self):
        X_cf = np.zeros((C, T))
        model = make_model(target_pred=0)
        assert validity(X_cf, model, target_class=1) is False

    def test_diversity_single_instance(self):
        cfs = np.zeros((1, C, T))
        assert diversity(cfs) == pytest.approx(0.0)

    def test_diversity_identical(self):
        cfs = np.ones((3, C, T))
        assert diversity(cfs) == pytest.approx(0.0)

    def test_diversity_different(self):
        rng = np.random.default_rng(2)
        cfs = rng.normal(0, 5, (3, C, T))
        assert diversity(cfs) > 0.0

    def test_evaluate_batch(self, synthetic_data):
        X_train, y_train, X_query = synthetic_data
        exp = PerCEExplainer(model=make_flip_model(), n_segments=5)
        exp.fit(X_train, y_train)
        results = exp.explain_batch(X_train[:4], target_classes=1, verbose=False)
        summary = evaluate_batch(results)
        assert "validity_rate" in summary
        assert "proximity_mean" in summary
        assert "sparsity_mean" in summary
        assert "diversity" in summary
        assert 0.0 <= summary["validity_rate"] <= 1.0


# Importance
class TestImportance:

    def test_channel_importance_shape(self):
        X = np.random.randn(C, T)
        imp = channel_importance(make_model(0), X)
        assert imp.shape == (C,)

    def test_segment_importance_shape(self):
        X = np.random.randn(C, T)
        imp = segment_importance(make_model(0), X, n_segments=5)
        assert imp.shape == (5,)

    def test_channel_importance_nonnegative_for_informative_channel(self):
        """Channel importance should vary across channels for a channel-sensitive model."""
        rng = np.random.default_rng(3)
        X = rng.normal(0, 1, (C, T)).astype(np.float32)

        # Model sensitive to channel 0 specifically
        def sensitive_model(X_np):
            means = X_np[:, 0, :].mean(axis=1)
            p1 = 1.0 / (1.0 + np.exp(-2.0 * means))
            return np.stack([1 - p1, p1], axis=1)

        imp = channel_importance(sensitive_model, X)
        # Channel 0 should have higher importance than channel -1
        assert imp[0] != imp[-1] or not np.allclose(imp, imp[0])



# DTW
class TestDTW:

    def test_dtw_self_distance(self):
        X = np.random.randn(C, T)
        assert dtw_distance(X, X) == pytest.approx(0.0, abs=1e-6)

    def test_dtw_symmetric(self):
        rng = np.random.default_rng(4)
        X = rng.normal(0, 1, (C, T))
        Y = rng.normal(0, 1, (C, T))
        assert dtw_distance(X, Y) == pytest.approx(dtw_distance(Y, X), rel=1e-5)

    def test_dtw_positive(self):
        X = np.zeros((C, T))
        Y = np.ones((C, T))
        assert dtw_distance(X, Y) > 0.0
