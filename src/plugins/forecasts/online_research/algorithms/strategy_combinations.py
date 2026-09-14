"""Causal strategy-combination experts for live prediction.

Every active strategy direction becomes a signed token. The learner maintains
bounded Beta-style experts for singles, pairs and triples of those tokens, then
blends only experts with enough matured evidence. This explores useful
combinations without attempting the impossible 2^N materialisation of every
strategy permutation.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

from .base import OnlineClassifier


class OnlineStrategyCombinations(OnlineClassifier):
    name = "strategy_combinations"

    def __init__(
        self,
        max_active: int = 8,
        max_order: int = 3,
        min_support: int = 4,
        min_strength: float = 0.15,
        prior_strength: float = 2.0,
        decay: float = 0.995,
        max_states: int = 5000,
    ) -> None:
        if max_active < 1 or not 1 <= max_order <= 3:
            raise ValueError("strategy combinations require max_active >= 1 and max_order in 1..3")
        if min_support < 1 or min_strength < 0 or prior_strength <= 0:
            raise ValueError("strategy combination support, strength and prior must be positive")
        if not 0.0 < decay <= 1.0 or max_states < 10:
            raise ValueError("strategy combination decay must be in (0, 1] and max_states >= 10")
        self.max_active = int(max_active)
        self.max_order = int(max_order)
        self.min_support = int(min_support)
        self.min_strength = float(min_strength)
        self.prior_strength = float(prior_strength)
        self.decay = float(decay)
        self.max_states = int(max_states)
        # key -> [up count, down count, last global sample]
        self.experts: dict[str, list[float]] = {}
        self.samples_seen = 0

    def predict_proba(self, features: dict[str, float]) -> float:
        estimates: list[tuple[float, float]] = []
        for key in self._keys(features):
            counts = self.experts.get(key)
            if counts is None:
                continue
            up, down = self._decayed(counts)
            support = up + down
            if support < self.min_support:
                continue
            probability = (up + self.prior_strength / 2.0) / (
                support + self.prior_strength
            )
            # More evidence matters, but sqrt prevents one old common state from
            # drowning out a sharper pair/triple that has enough observations.
            estimates.append((probability, math.sqrt(support)))
        if not estimates:
            return 0.5
        total = sum(weight for _, weight in estimates)
        probability = sum(value * weight for value, weight in estimates) / total
        return max(1e-6, min(1.0 - 1e-6, probability))

    def update(self, features: dict[str, float], target: int) -> None:
        selected = 1 if target else 0
        for key in self._keys(features):
            counts = self.experts.get(key, [0.0, 0.0, float(self.samples_seen)])
            up, down = self._decayed(counts)
            if selected:
                up += 1.0
            else:
                down += 1.0
            self.experts[key] = [up, down, float(self.samples_seen)]
        self.samples_seen += 1
        self._prune()

    def _keys(self, features: dict[str, float]) -> tuple[str, ...]:
        active = []
        for name, value in features.items():
            if not name.startswith("strategy__") or not math.isfinite(value):
                continue
            strength = abs(float(value))
            if strength < self.min_strength:
                continue
            strategy = name.removeprefix("strategy__")
            token = f"{strategy}{'+' if value > 0 else '-'}"
            active.append((strength, token))
        tokens = [token for _, token in sorted(active, key=lambda item: (-item[0], item[1]))]
        tokens = sorted(tokens[: self.max_active])
        keys = []
        for order in range(1, min(self.max_order, len(tokens)) + 1):
            keys.extend(f"{order}|{'&'.join(items)}" for items in itertools.combinations(tokens, order))
        return tuple(keys)

    def _decayed(self, counts: list[float]) -> tuple[float, float]:
        elapsed = max(self.samples_seen - int(counts[2]), 0)
        factor = self.decay**elapsed
        return float(counts[0]) * factor, float(counts[1]) * factor

    def _prune(self) -> None:
        if len(self.experts) <= self.max_states:
            return
        ranked = sorted(
            self.experts,
            key=lambda key: sum(self._decayed(self.experts[key])),
            reverse=True,
        )
        self.experts = {key: self.experts[key] for key in ranked[: self.max_states]}

    def state_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "max_active": self.max_active,
            "max_order": self.max_order,
            "min_support": self.min_support,
            "min_strength": self.min_strength,
            "prior_strength": self.prior_strength,
            "decay": self.decay,
            "max_states": self.max_states,
            "samples_seen": self.samples_seen,
            "experts": {key: list(counts) for key, counts in self.experts.items()},
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.max_active = int(payload.get("max_active", self.max_active))
        self.max_order = int(payload.get("max_order", self.max_order))
        self.min_support = int(payload.get("min_support", self.min_support))
        self.min_strength = float(payload.get("min_strength", self.min_strength))
        self.prior_strength = float(payload.get("prior_strength", self.prior_strength))
        self.decay = float(payload.get("decay", self.decay))
        self.max_states = int(payload.get("max_states", self.max_states))
        self.samples_seen = int(payload.get("samples_seen", 0))
        self.experts = {
            str(key): [float(values[0]), float(values[1]), float(values[2])]
            for key, values in dict(payload.get("experts", {})).items()
            if len(values) == 3
        }
        self._prune()


__all__ = ["OnlineStrategyCombinations"]
