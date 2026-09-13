"""Probability calibration.

A boosted tree's "0.7" is a score, not a probability. Gradient boosting optimises
log loss and produces overconfident, often bimodal outputs, so a bar scored 0.7
may historically resolve upward only 54% of the time. Since the strategy sizes
positions from probability, that gap matters directly.

Two design choices keep calibration from doing more harm than good:

**Platt scaling is the default, not isotonic.** Isotonic regression is
non-parametric and can fit any monotone shape, which sounds strictly better until
you use it on weak signal. There it collapses most inputs to nearly the same
output — on this project's data it mapped 85% of predictions into a single bucket
at 0.50, destroying the ranking resolution downstream code depends on. Platt is a
one-parameter logistic fit: it cannot collapse, it degrades gracefully, and it
strictly preserves ordering.

**Calibration is applied only when it demonstrably helps.** The fit is validated
on a chronological holdout of the out-of-sample predictions and rejected unless
it improves the Brier score. A calibrator that does not help should do nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


@dataclass
class ReliabilityBin:
    lower: float
    upper: float
    predicted: float
    observed: float
    samples: int


class ProbabilityCalibrator:
    """Maps raw model scores onto observed frequencies."""

    def __init__(
        self,
        method: str = "platt",
        min_samples: int = 400,
        validation_fraction: float = 0.3,
        require_improvement: bool = True,
        min_improvement_ratio: float = 0.005,
        min_spread_ratio: float = 0.10,
    ) -> None:
        if method not in {"platt", "isotonic", "none"}:
            raise ValueError(f"unknown calibration method {method!r}")
        self.method = method
        self.min_samples = min_samples
        self.validation_fraction = validation_fraction
        self.require_improvement = require_improvement
        # Required relative gain in Brier score. Testing on already-calibrated
        # input shows a one-parameter fit still nudges the holdout Brier by
        # roughly 1e-5 in either direction — pure noise. Accepting on any
        # improvement at all would therefore accept noise half the time, so the
        # gate demands a gain that is large relative to the score itself.
        self.min_improvement_ratio = min_improvement_ratio
        # Minimum retained standard deviation, as a fraction of the raw spread —
        # that is, the largest shrinkage tolerated before calibration is treated
        # as having replaced the model with a constant. Measured on synthetic
        # data, a model with no signal at all shrinks by ~140x (ratio 0.007),
        # while a genuinely informative model that is 2x, 3x, or 5x overconfident
        # shrinks by 1.9x, 3.0x, and 4.9x (ratios 0.51, 0.34, 0.21). A threshold
        # of 0.10 sits well clear of both groups.
        self.min_spread_ratio = min_spread_ratio
        self._model = None
        self._active = False
        self.reason = "not fitted"
        self.validation = {}

    def fit(self, probabilities: np.ndarray, y_true: np.ndarray) -> ProbabilityCalibrator:
        probabilities = np.asarray(probabilities, dtype="float64").ravel()
        y_true = np.asarray(y_true).astype(int).ravel()

        if self.method == "none":
            self.reason = "calibration disabled"
            return self
        if len(probabilities) < self.min_samples:
            self.reason = f"only {len(probabilities)} samples (need {self.min_samples})"
            return self
        if len(np.unique(y_true)) < 2:
            self.reason = "labels are single-class"
            return self

        # Chronological split: calibrating on later data than you validate on
        # would leak the future into the map.
        cut = int(len(probabilities) * (1.0 - self.validation_fraction))
        fit_p, fit_y = probabilities[:cut], y_true[:cut]
        val_p, val_y = probabilities[cut:], y_true[cut:]

        if len(np.unique(fit_y)) < 2 or len(val_p) < 50:
            self.reason = "insufficient training split"
            return self

        candidate = self._build(self.method)
        self._fit_model(candidate, fit_p, fit_y)

        calibrated = self._apply(candidate, val_p)
        calibrated_full = self._apply(candidate, probabilities)
        raw_brier = brier_score_loss(val_y, np.clip(val_p, 1e-6, 1 - 1e-6))
        cal_brier = brier_score_loss(val_y, np.clip(calibrated, 1e-6, 1 - 1e-6))
        improvement = raw_brier - cal_brier
        required = raw_brier * self.min_improvement_ratio

        self.validation = {
            "raw_brier": float(raw_brier),
            "calibrated_brier": float(cal_brier),
            "improvement": float(improvement),
            "required": float(required),
        }

        if self.require_improvement and improvement < required:
            self.reason = (
                f"rejected: Brier gain {improvement:+.6f} below required {required:.6f}"
            )
            return self

        # A Brier gain alone is not enough to justify calibrating. When the model
        # has no real signal, mapping everything toward the base rate always
        # improves Brier — because the honest probability really is the base rate
        # — while flattening the distribution to near-constant. That looks like a
        # win on the score while destroying the score spread a threshold-based
        # strategy needs to act on. If calibration shrinks the model's opinion
        # this hard, it has stopped being a calibration and become a constant
        # predictor; the base rate is already reported separately, so reject it
        # and leave the model's own ranking intact.
        raw_spread = float(np.std(probabilities))
        cal_spread = float(np.std(calibrated_full))
        spread_ratio = cal_spread / raw_spread if raw_spread > 0 else 1.0
        self.validation["raw_spread"] = raw_spread
        self.validation["calibrated_spread"] = cal_spread
        self.validation["spread_ratio"] = spread_ratio

        if spread_ratio < self.min_spread_ratio:
            self.reason = (
                f"rejected: would collapse score spread to {spread_ratio:.1%} of raw "
                f"(Brier gain {improvement:+.6f})"
            )
            return self

        # Refit on everything now that the method has earned its place.
        final = self._build(self.method)
        self._fit_model(final, probabilities, y_true)
        self._model = final
        self._active = True
        self.reason = (
            f"{self.method} active, Brier {raw_brier:.5f} -> {cal_brier:.5f} on holdout"
        )
        return self

    def _build(self, method: str):
        if method == "isotonic":
            return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        return LogisticRegression(C=1.0, max_iter=1000)

    def _fit_model(self, model, probabilities: np.ndarray, y_true: np.ndarray) -> None:
        if self.method == "isotonic":
            model.fit(probabilities, y_true)
        else:
            model.fit(_logit(probabilities).reshape(-1, 1), y_true)

    def _apply(self, model, probabilities: np.ndarray) -> np.ndarray:
        if self.method == "isotonic":
            return np.asarray(model.predict(probabilities), dtype="float64")
        return model.predict_proba(_logit(probabilities).reshape(-1, 1))[:, 1]

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(probabilities, dtype="float64").ravel()
        if not self._active or self._model is None:
            return probabilities
        calibrated = self._apply(self._model, probabilities)
        return np.clip(calibrated, 1e-6, 1 - 1e-6)

    @property
    def is_fitted(self) -> bool:
        """True when calibration is actually being applied."""
        return self._active

    def describe(self) -> str:
        return self.reason


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def reliability_table(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    bins: int = 10,
) -> list[ReliabilityBin]:
    """Predicted vs observed frequency per probability bucket."""
    probabilities = np.asarray(probabilities, dtype="float64").ravel()
    y_true = np.asarray(y_true).astype(int).ravel()
    if len(probabilities) == 0:
        return []

    edges = np.linspace(0.0, 1.0, bins + 1)
    out: list[ReliabilityBin] = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=False):
        mask = (probabilities >= lower) & (probabilities < upper)
        if mask.sum() == 0:
            continue
        out.append(
            ReliabilityBin(
                lower=float(lower),
                upper=float(upper),
                predicted=float(probabilities[mask].mean()),
                observed=float(y_true[mask].mean()),
                samples=int(mask.sum()),
            )
        )
    return out


def expected_calibration_error(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    bins: int = 10,
) -> float:
    """Sample-weighted mean gap between predicted and observed frequency."""
    table = reliability_table(probabilities, y_true, bins=bins)
    if not table:
        return float("nan")
    total = sum(entry.samples for entry in table)
    return float(
        sum(entry.samples * abs(entry.predicted - entry.observed) for entry in table) / total
    )
