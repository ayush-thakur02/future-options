"""Incremental logistic regression optimised by one-sample SGD."""

from __future__ import annotations

from typing import Any

from .base import ScaledLinearClassifier, sigmoid


class OnlineLogistic(ScaledLinearClassifier):
    name = "online_logistic"

    def __init__(self, learning_rate: float = 0.04, l2: float = 0.0001) -> None:
        super().__init__()
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)

    def predict_proba(self, features: dict[str, float]) -> float:
        return sigmoid(self.margin(features))

    def update(self, features: dict[str, float], target: int) -> None:
        scaled = self.scaler.transform(features)
        probability = sigmoid(
            self.bias
            + sum(self.weights.get(name, 0.0) * value for name, value in scaled.items())
        )
        error = float(target) - probability
        # A decaying step keeps long-running state stable while retaining the
        # ability to adapt after a regime change.
        step = self.learning_rate / (1.0 + self.samples_seen * 0.0005) ** 0.5
        shrink = max(0.0, 1.0 - step * self.l2)
        for name, value in scaled.items():
            self.weights[name] = self.weights.get(name, 0.0) * shrink + step * error * value
        self.bias += step * error
        self.samples_seen += 1
        self.scaler.update(features)

    def _parameters(self) -> dict[str, Any]:
        return {"learning_rate": self.learning_rate, "l2": self.l2}

    def _load_parameters(self, payload: dict[str, Any]) -> None:
        self.learning_rate = float(payload.get("learning_rate", self.learning_rate))
        self.l2 = float(payload.get("l2", self.l2))


__all__ = ["OnlineLogistic"]
