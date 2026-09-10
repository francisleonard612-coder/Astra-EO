from __future__ import annotations

import numpy as np

from models.base import N_DIGITS, normalize


def combine(predictions: dict[str, np.ndarray], weights: dict[str, float | np.ndarray]) -> np.ndarray:
    """Weighted average of model probability vectors.

    `weights[model]` may be a single scalar (one weight applied to that
    model's whole vector -- the original ensemble behavior) or a length-10
    array (a separate weight per digit -- per-digit specialist weighting,
    see learning/online.py). Both are supported transparently: a scalar is
    just broadcast across all 10 digits. Missing/zero weights are skipped;
    the result is always renormalized to sum to 1.
    """
    acc = np.zeros(N_DIGITS)
    total_weight = np.zeros(N_DIGITS)
    for name, vec in predictions.items():
        w = weights.get(name, 0.0)
        w_arr = np.full(N_DIGITS, w, dtype=float) if np.isscalar(w) else np.asarray(w, dtype=float)
        acc += w_arr * vec
        total_weight += w_arr
    if not np.any(total_weight > 0):
        return np.full(N_DIGITS, 1.0 / N_DIGITS)
    # guard any individual digit whose total weight is zero (all contributing
    # models had zero weight for that digit) so division stays well-defined
    safe_total = np.where(total_weight > 0, total_weight, 1.0)
    combined = np.where(total_weight > 0, acc / safe_total, 1.0 / N_DIGITS)
    return normalize(combined)


EVEN_DIGITS = (0, 2, 4, 6, 8)
ODD_DIGITS = (1, 3, 5, 7, 9)


def model_agreement(predictions: dict[str, np.ndarray],
                     weights: dict[str, float | np.ndarray] | None = None) -> dict[str, float]:
    """(Optionally weighted) standard deviation across models of P(Even) and
    P(Odd). Lower stdev = higher agreement. Returned as a normalized [0, 1]
    "agreement score" (1 = perfect agreement) for both contract sides.

    Pass `weights` -- the SAME per-model trust weights `combine()` actually
    uses to build the traded ensemble (e.g. SymbolPipeline.champion_weights)
    -- to compute agreement the ensemble's own weighting, not a naive
    unweighted std. Without this, a single model the ensemble has already
    learned to all-but-ignore (e.g. a ~2% blend weight) can still swing an
    unweighted std enough to look like "the models are in wild disagreement"
    -- which upstream (RegimeDetector, pricing/mispricing.py's
    "model_disagreement" gate) can veto every trade even while the
    properly-weighted, ACTUALLY-TRADED probability is calm and confident.
    This was caught from a live deployment: an online SGD sub-model
    (models/sklearn_models.py::LogisticModel) was regularly emitting
    near-one-hot, tick-to-tick-unstable predictions (a known failure mode of
    single-sample partial_fit with little regularization) that the ensemble
    had already down-weighted to ~2%, yet an unweighted std let it alone
    trip MODEL_DISAGREEMENT on almost every tick.

    A scalar or length-10 per-digit weight per model is accepted (arrays
    are reduced to their mean for this purpose, since agreement here is
    scored per contract side, not per digit). A model missing from
    `weights` falls back to equal weighting so it's never silently dropped,
    and if `weights` is omitted entirely (or nets to all-zero), this
    degrades transparently to the original unweighted behavior.
    """
    names = list(predictions.keys())
    even_vals = np.array([float(np.sum(v[list(EVEN_DIGITS)])) for v in predictions.values()])
    odd_vals = np.array([float(np.sum(v[list(ODD_DIGITS)])) for v in predictions.values()])

    if weights:
        w = np.array([
            float(np.mean(weights[name])) if name in weights else 1.0
            for name in names
        ])
        if not np.any(w > 0):
            w = np.ones(len(names))
    else:
        w = np.ones(len(names))

    even_std = _weighted_std(even_vals, w) if len(even_vals) > 1 else 0.0
    odd_std = _weighted_std(odd_vals, w) if len(odd_vals) > 1 else 0.0
    # a stdev of 0.25 across models on a probability in [0,1] is already very
    # high disagreement; use that as the scale for a 0..1 "agreement" score
    scale = 0.25
    return {
        "even_agreement": max(0.0, 1.0 - min(even_std / scale, 1.0)),
        "odd_agreement": max(0.0, 1.0 - min(odd_std / scale, 1.0)),
        "even_std": even_std,
        "odd_std": odd_std,
    }


def _weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    if total <= 0:
        return float(np.std(values))
    w = weights / total
    mean = float(np.sum(w * values))
    variance = float(np.sum(w * (values - mean) ** 2))
    return float(np.sqrt(max(variance, 0.0)))
