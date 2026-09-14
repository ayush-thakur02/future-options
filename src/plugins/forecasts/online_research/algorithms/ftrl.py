"""FTRL-Proximal: sparse adaptive online logistic regression.

FTRL keeps one learning-rate accumulator per feature, so noisy indicators are
automatically given smaller steps while new signals can still adapt quickly.
The proximal L1 term also removes persistently unhelpful coefficients instead
of letting a 100+ column feature set accumulate noise forever.
"""

from __future__ import annotations

import math
from typing import Any

from .base import CausalScaler, OnlineClassifier, sigmoid


class OnlineFTRL(OnlineClassifier):
    name = "ftrl_proximal"

    def __init__(
        self,
        alpha: float = 0.08,
        beta: float = 1.0,
        l1: float = 0.002,
        l2: float = 0.01,
    ) -> None:
        if alpha <= 0 or beta < 0 or l1 < 0 or l2 < 0:
            raise ValueError("FTRL requires alpha > 0 and non-negative beta/l1/l2")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.l1 = float(l1)
        self.l2 = float(l2)
        self.z: dict[str, float] = {}
        self.n: dict[str, float] = {}
        self.bias_z = 0.0
        self.bias_n = 0.0
        self.samples_seen = 0
        self.scaler = CausalScaler()

    def _weight(self, z: float, n: float, *, regularise: bool = True) -> float:
        l1 = self.l1 if regularise else 0.0
        if abs(z) <= l1:
            return 0.0
        denominator = (self.beta + math.sqrt(n)) / self.alpha + self.l2
        return -(z - math.copysign(l1, z)) / denominator

    def predict_proba(self, features: dict[str, float]) -> float:
        scaled = self.scaler.transform(features)
        margin = self._weight(self.bias_z, self.bias_n, regularise=False)
        margin += sum(
            self._weight(self.z.get(name, 0.0), self.n.get(name, 0.0)) * value
            for name, value in scaled.items()
        )
        return sigmoid(margin)

    def update(self, features: dict[str, float], target: int) -> None:
        scaled = self.scaler.transform(features)
        probability = self.predict_proba(features)
        error = probability - float(bool(target))

        bias_weight = self._weight(self.bias_z, self.bias_n, regularise=False)
        bias_next = self.bias_n + error * error
        sigma = (math.sqrt(bias_next) - math.sqrt(self.bias_n)) / self.alpha
        self.bias_z += error - sigma * bias_weight
        self.bias_n = bias_next

        for name, value in scaled.items():
            gradient = error * value
            old_n = self.n.get(name, 0.0)
            new_n = old_n + gradient * gradient
            weight = self._weight(self.z.get(name, 0.0), old_n)
            sigma = (math.sqrt(new_n) - math.sqrt(old_n)) / self.alpha
            self.z[name] = self.z.get(name, 0.0) + gradient - sigma * weight
            self.n[name] = new_n

        self.samples_seen += 1
        self.scaler.update(features)

    def state_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "alpha": self.alpha,
            "beta": self.beta,
            "l1": self.l1,
            "l2": self.l2,
            "z": dict(self.z),
            "n": dict(self.n),
            "bias_z": self.bias_z,
            "bias_n": self.bias_n,
            "samples_seen": self.samples_seen,
            "scaler": self.scaler.as_dict(),
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.alpha = float(payload.get("alpha", self.alpha))
        self.beta = float(payload.get("beta", self.beta))
        self.l1 = float(payload.get("l1", self.l1))
        self.l2 = float(payload.get("l2", self.l2))
        self.z = {str(name): float(value) for name, value in dict(payload.get("z", {})).items()}
        self.n = {str(name): float(value) for name, value in dict(payload.get("n", {})).items()}
        self.bias_z = float(payload.get("bias_z", 0.0))
        self.bias_n = float(payload.get("bias_n", 0.0))
        self.samples_seen = int(payload.get("samples_seen", 0))
        self.scaler = CausalScaler.from_dict(dict(payload.get("scaler", {})))


__all__ = ["OnlineFTRL"]
