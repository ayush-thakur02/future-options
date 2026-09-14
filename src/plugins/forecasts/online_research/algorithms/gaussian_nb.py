"""Incremental Gaussian Naive Bayes with Welford class statistics."""

from __future__ import annotations

import math
from typing import Any

from .base import OnlineClassifier, RunningMoment, sigmoid


class OnlineGaussianNB(OnlineClassifier):
    name = "gaussian_nb"

    def __init__(self, variance_floor: float = 1e-6) -> None:
        self.variance_floor = float(variance_floor)
        self.class_count = [0, 0]
        self.moments: list[dict[str, RunningMoment]] = [{}, {}]
        self.samples_seen = 0

    def predict_proba(self, features: dict[str, float]) -> float:
        total = sum(self.class_count)
        log_scores: list[float] = []
        for target in (0, 1):
            score = math.log((self.class_count[target] + 1.0) / (total + 2.0))
            for name, value in features.items():
                moment = self.moments[target].get(name)
                if moment is None or moment.count < 2:
                    continue
                variance = max(moment.standard_deviation**2, self.variance_floor)
                score += -0.5 * (math.log(2.0 * math.pi * variance) + (value - moment.mean) ** 2 / variance)
            log_scores.append(score)
        # Only the log-score difference matters and sigmoid avoids exponent overflow.
        return sigmoid(max(-60.0, min(60.0, log_scores[1] - log_scores[0])))

    def update(self, features: dict[str, float], target: int) -> None:
        selected = 1 if target else 0
        self.class_count[selected] += 1
        for name, value in features.items():
            self.moments[selected].setdefault(name, RunningMoment()).update(value)
        self.samples_seen += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "variance_floor": self.variance_floor,
            "class_count": list(self.class_count),
            "samples_seen": self.samples_seen,
            "moments": [
                {name: moment.as_dict() for name, moment in class_moments.items()}
                for class_moments in self.moments
            ],
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.variance_floor = float(payload.get("variance_floor", self.variance_floor))
        counts = list(payload.get("class_count", [0, 0]))
        self.class_count = [int(counts[0]), int(counts[1])]
        self.samples_seen = int(payload.get("samples_seen", sum(self.class_count)))
        saved = list(payload.get("moments", [{}, {}]))
        self.moments = [
            {
                str(name): RunningMoment.from_dict(dict(moment))
                for name, moment in dict(saved[target]).items()
            }
            for target in (0, 1)
        ]


__all__ = ["OnlineGaussianNB"]
