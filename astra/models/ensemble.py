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


def model_agreement(predictions: dict[str, np.ndarray]) -> dict[str, float]:
    """Standard deviation across models of P(Even) and P(Odd).
    Lower stdev = higher agreement. Returned as a normalized [0, 1]
    "agreement score" (1 = perfect agreement) for both contract sides."""
    even_vals = [float(np.sum(v[list(EVEN_DIGITS)])) for v in predictions.values()]
    odd_vals = [float(np.sum(v[list(ODD_DIGITS)])) for v in predictions.values()]
    even_std = float(np.std(even_vals)) if len(even_vals) > 1 else 0.0
    odd_std = float(np.std(odd_vals)) if len(odd_vals) > 1 else 0.0
    # a stdev of 0.25 across models on a probability in [0,1] is already very
    # high disagreement; use that as the scale for a 0..1 "agreement" score
    scale = 0.25
    return {
        "even_agreement": max(0.0, 1.0 - min(even_std / scale, 1.0)),
        "odd_agreement": max(0.0, 1.0 - min(odd_std / scale, 1.0)),
        "even_std": even_std,
        "odd_std": odd_std,
    }
