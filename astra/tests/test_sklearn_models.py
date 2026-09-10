import numpy as np

from features.feature_engine import build_features
from models.sklearn_models import LogisticModel
from state.rolling_state import StateManager

WINDOWS = [20, 50, 100, 250, 500, 1000]


def _run_on_random_digits(model, n=600, seed=1):
    rng = np.random.default_rng(seed)
    sm = StateManager(max_window=500, max_markov_order=3)
    state = sm.get("SIM")
    max_probas = []
    for _ in range(n):
        d = int(rng.integers(0, 10))
        bundle = build_features(state, windows=WINDOWS)
        if model.is_ready(state):
            vec = model.predict(state, bundle)
            assert abs(vec.sum() - 1.0) < 1e-6
            assert (vec >= 0).all()
            max_probas.append(float(vec.max()))
        model.observe(state, bundle, d)
        state.push(d)
    return max_probas


def test_logistic_model_does_not_collapse_to_near_one_hot_on_random_digits():
    """Regression test for a live-deployment bug: the original config
    (alpha=1e-4, no weight averaging) put predict()'s max-probability above
    0.99 on ~98% of ticks when fed a genuinely random digit stream -- a
    single-sample-SGD instability, not a real learned signal. With
    average=True and a larger alpha, that should no longer happen."""
    model = LogisticModel()
    max_probas = _run_on_random_digits(model)
    tail = max_probas[-300:]
    assert tail, "model should have started predicting well before tick 300"
    frac_saturated = sum(1 for p in tail if p > 0.99) / len(tail)
    assert frac_saturated < 0.1
    assert sum(tail) / len(tail) < 0.7


def test_logistic_model_output_is_always_a_valid_clipped_distribution():
    model = LogisticModel()
    max_probas = _run_on_random_digits(model, n=200)
    assert all(0.0 < p < 1.0 for p in max_probas)


def test_logistic_model_predicts_uniform_before_fitted():
    model = LogisticModel()
    sm = StateManager(max_window=500, max_markov_order=3)
    state = sm.get("SIM")
    state.push(3)
    bundle = build_features(state, windows=WINDOWS)
    vec = model.predict(state, bundle)
    assert np.allclose(vec, np.full(10, 0.1))
