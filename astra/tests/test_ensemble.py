import numpy as np

from models.ensemble import combine, model_agreement


def test_combine_weighted_average_sums_to_one():
    preds = {
        "a": np.array([0.1] * 10),
        "b": np.array([0.05] * 5 + [0.15] * 5),
    }
    weights = {"a": 0.5, "b": 0.5}
    result = combine(preds, weights)
    assert abs(result.sum() - 1.0) < 1e-9


def test_combine_ignores_zero_weight_models():
    preds = {
        "a": np.full(10, 0.1),
        "b": np.array([1.0] + [0.0] * 9),  # would badly skew result if included
    }
    weights = {"a": 1.0, "b": 0.0}
    result = combine(preds, weights)
    assert np.allclose(result, np.full(10, 0.1), atol=1e-6)


def test_model_agreement_high_when_models_agree():
    preds = {
        "a": np.array([0.05] * 3 + [0.55] + [0.05] * 6),
        "b": np.array([0.05] * 3 + [0.55] + [0.05] * 6),
    }
    agreement = model_agreement(preds)
    assert agreement["even_agreement"] > 0.95
    assert agreement["odd_agreement"] > 0.95


def test_model_agreement_low_when_models_disagree():
    preds = {
        "a": np.array([0.0] * 9 + [1.0]),   # thinks digit is always 9 (odd) -> high P(odd)
        "b": np.array([1.0] + [0.0] * 9),   # thinks digit is always 0 (even) -> high P(even)
    }
    agreement = model_agreement(preds)
    assert agreement["even_agreement"] < 0.5
    assert agreement["odd_agreement"] < 0.5


def test_model_agreement_downweights_a_distrusted_outlier_model():
    """A model the ensemble has already learned to distrust (tiny blend
    weight) shouldn't be able to single-handedly make well-agreeing, highly-
    trusted models look like they're in wild disagreement."""
    preds = {
        "trusted_a": np.array([0.05] * 3 + [0.55] + [0.05] * 6),
        "trusted_b": np.array([0.06] * 3 + [0.52] + [0.06] * 6),
        "distrusted_outlier": np.array([0.0] * 9 + [1.0]),
    }
    weights = {"trusted_a": 0.49, "trusted_b": 0.49, "distrusted_outlier": 0.02}

    weighted = model_agreement(preds, weights=weights)
    unweighted = model_agreement(preds)

    assert weighted["even_agreement"] > unweighted["even_agreement"]
    assert weighted["odd_agreement"] > unweighted["odd_agreement"]
    # the trusted pair agrees almost perfectly, so weighted agreement should
    # end up high once the outlier's influence is properly discounted
    assert weighted["even_agreement"] > 0.7


def test_model_agreement_missing_weight_falls_back_to_equal_weighting():
    """A model with no entry in `weights` shouldn't be silently dropped --
    it should count as if equally weighted, same as the unweighted path."""
    preds = {
        "a": np.array([0.05] * 3 + [0.55] + [0.05] * 6),
        "b": np.array([0.05] * 3 + [0.55] + [0.05] * 6),
    }
    partial_weights = {"a": 0.9}  # "b" intentionally omitted
    agreement = model_agreement(preds, weights=partial_weights)
    assert agreement["even_agreement"] > 0.95


def test_model_agreement_accepts_per_digit_weight_arrays():
    """Champion weights are length-10 per-digit arrays, not scalars --
    model_agreement should reduce them to a representative trust weight
    (their mean) rather than choking on the shape."""
    preds = {
        "trusted": np.array([0.05] * 3 + [0.55] + [0.05] * 6),
        "distrusted_outlier": np.array([0.0] * 9 + [1.0]),
    }
    weights = {
        "trusted": np.full(10, 0.9),
        "distrusted_outlier": np.full(10, 0.02),
    }
    weighted = model_agreement(preds, weights=weights)
    unweighted = model_agreement(preds)
    assert weighted["even_agreement"] > unweighted["even_agreement"]
