"""
Architecture B: 10 independent digit specialists.

Each specialist estimates P(digit == d) as its OWN binary classification
problem -- a genuinely different modeling philosophy from Architecture A
(models/registry.py's shared multiclass models with adaptive per-digit
weighting), not just a relabeling of the same computation. Per spec
sections 5 & 7.

SCOPE NOTE: each specialist is a small 2-signal ensemble (a Beta-Bernoulli
conjugate estimate + an online binary logistic regression), not the full
model zoo (Bayesian+Markov+RF+XGBoost etc.) duplicated 10 times over. That
would be 10x the heaviest models' memory/compute footprint per symbol,
which doesn't scale to "room for more symbols" per-instance. This is a
deliberate trade-off, not an oversight -- it's still a materially different
architecture from A (see the docstring on ArchitectureCompetitionManager
for how the two are actually compared, empirically, rather than assumed).

The 10 independent P(digit=d) estimates don't sum to 1 by construction
(each specialist has no knowledge of the others), so they're combined into
a valid distribution by renormalization: p_d / sum(p). This is the standard
way to turn a set of one-vs-rest binary classifiers into a multiclass
distribution, and each specialist's own calibration (tracked upstream by
ArchitectureCompetitionManager) can correct systematic bias in what it
outputs before that renormalization happens.
"""
from __future__ import annotations

import numpy as np

from features.feature_engine import FeatureBundle
from state.rolling_state import SymbolState

N_DIGITS = 10


class DigitSpecialist:
    """One specialist, dedicated to estimating P(digit == self.digit)."""

    def __init__(self, digit: int):
        self.digit = digit
        # Beta-Bernoulli prior centered on the digit's fair unconditional
        # rate (1/10): alpha=1, beta=9 gives a prior mean of exactly 0.1.
        self._alpha = 1.0
        self._beta = 9.0
        self._fitted = False
        self._logistic = None  # constructed lazily to avoid importing sklearn at module load for unused digits

    def _ensure_logistic(self):
        if self._logistic is None:
            from sklearn.linear_model import SGDClassifier
            self._logistic = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=1, warm_start=True)

    def bayes_probability(self) -> float:
        return self._alpha / (self._alpha + self._beta)

    def logistic_probability(self, feature_vector: np.ndarray | None) -> float | None:
        if not self._fitted or feature_vector is None:
            return None
        self._ensure_logistic()
        try:
            proba = self._logistic.predict_proba(feature_vector.reshape(1, -1))[0]
            classes = self._logistic.classes_
            return float(proba[list(classes).index(1)]) if 1 in classes else 0.0
        except Exception:  # noqa: BLE001
            return None

    def predict_probability(self, feature_vector: np.ndarray | None) -> float:
        bayes_p = self.bayes_probability()
        logit_p = self.logistic_probability(feature_vector)
        if logit_p is None:
            return bayes_p
        return 0.5 * bayes_p + 0.5 * logit_p

    def observe(self, feature_vector: np.ndarray | None, actual_digit: int) -> None:
        y = 1 if actual_digit == self.digit else 0
        if y:
            self._alpha += 1
        else:
            self._beta += 1

        if feature_vector is None:
            return
        self._ensure_logistic()
        x = feature_vector.reshape(1, -1)
        y_arr = np.array([y])
        try:
            if not self._fitted:
                self._logistic.partial_fit(x, y_arr, classes=np.array([0, 1]))
                self._fitted = True
            else:
                self._logistic.partial_fit(x, y_arr)
        except Exception:  # noqa: BLE001
            pass


def _renormalize(vec: np.ndarray) -> np.ndarray:
    vec = np.clip(vec, 1e-9, None)
    return vec / vec.sum()


class DigitSpecialistArchitecture:
    """Owns all 10 DigitSpecialist objects for one symbol and combines their
    independent P(digit=d) estimates into a single 10-digit distribution."""
    name = "specialist"

    def __init__(self):
        self.specialists: list[DigitSpecialist] = [DigitSpecialist(d) for d in range(N_DIGITS)]

    def predict_components(self, bundle: FeatureBundle) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (bayes_only_vector, logistic_only_vector, blended_vector) --
        the two sub-signal vectors are exposed separately so
        ArchitectureCompetitionManager can measure internal agreement
        between them, the same way Architecture A's agreement is measured
        across its 8 shared models."""
        bayes_vec = np.array([s.bayes_probability() for s in self.specialists])
        logit_raw = [s.logistic_probability(bundle.vector) for s in self.specialists]
        # fall back to bayes for any specialist not yet fitted
        logit_vec = np.array([lv if lv is not None else bv for lv, bv in zip(logit_raw, bayes_vec)])
        blended = np.array([s.predict_probability(bundle.vector) for s in self.specialists])
        return _renormalize(bayes_vec), _renormalize(logit_vec), _renormalize(blended)

    def predict(self, state: SymbolState, bundle: FeatureBundle) -> np.ndarray:
        _, _, blended = self.predict_components(bundle)
        return blended

    def observe(self, bundle: FeatureBundle, actual_digit: int) -> None:
        for s in self.specialists:
            s.observe(bundle.vector, actual_digit)

    def is_ready(self, state: SymbolState) -> bool:
        return state.total_observed >= 30
