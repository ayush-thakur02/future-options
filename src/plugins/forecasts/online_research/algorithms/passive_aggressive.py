"""Passive-aggressive online margin classifier."""

from __future__ import annotations

import math
from typing import Any

from .base import ScaledLinearClassifier, sigmoid


class OnlinePassiveAggressive(ScaledLinearClassifier):
    """PA-I updates only when the current sample violates the unit margin."""

    name = "passive_aggressive"

    def __init__(self, aggressiveness: float = 0.2) -> None:
        super().__init__()
        self.aggressiveness = float(aggressiveness)

    def predict_proba(self, features: dict[str, float]) -> float:
        raw_margin = self.margin(features)
        weight_norm = math.sqrt(sum(weight * weight for weight in self.weights.values()))
        scale = max(1.0, weight_norm)
        return sigmoid(raw_margin / scale)

    def update(self, features: dict[str, float], target: int) -> None:
        scaled = self.scaler.transform(features)
        signed_target = 1.0 if target else -1.0
        margin = self.bias + sum(
            self.weights.get(name, 0.0) * value for name, value in scaled.items()
        )
        loss = max(0.0, 1.0 - signed_target * margin)
        norm = 1.0 + sum(value * value for value in scaled.values())
        if loss > 0.0:
            step = min(self.aggressiveness, loss / norm) * signed_target
            for name, value in scaled.items():
                self.weights[name] = self.weights.get(name, 0.0) + step * value
            self.bias += step
        self.samples_seen += 1
        self.scaler.update(features)

    def _parameters(self) -> dict[str, Any]:
        return {"aggressiveness": self.aggressiveness}

    def _load_parameters(self, payload: dict[str, Any]) -> None:
        self.aggressiveness = float(payload.get("aggressiveness", self.aggressiveness))


__all__ = ["OnlinePassiveAggressive"]
