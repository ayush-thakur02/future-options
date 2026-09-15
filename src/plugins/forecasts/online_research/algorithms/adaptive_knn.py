"""Recency-weighted online k-nearest-neighbour classifier.

Unlike the linear online learners, this model can recognise a local market
state without forcing the same relationship to hold across every regime. It
keeps only a bounded rolling window and standardises exclusively with matured
observations, preserving the research lab's causal boundary.
"""

from __future__ import annotations

import math
from typing import Any

from .base import CausalScaler, OnlineClassifier


class OnlineAdaptiveKNN(OnlineClassifier):
    name = "adaptive_knn"

    def __init__(
        self,
        k: int = 11,
        window: int = 160,
        half_life: float = 80.0,
        max_features: int = 48,
        prior_strength: float = 2.0,
    ) -> None:
        if k < 1 or window < 2 or half_life <= 0 or max_features < 1 or prior_strength < 0:
            raise ValueError("adaptive KNN parameters must be positive")
        self.k = int(k)
        self.window = int(window)
        self.half_life = float(half_life)
        self.max_features = int(max_features)
        self.prior_strength = float(prior_strength)
        self.samples: list[tuple[dict[str, float], int]] = []
        self.samples_seen = 0
        self.scaler = CausalScaler()

    def predict_proba(self, features: dict[str, float]) -> float:
        if not self.samples:
            return 0.5
        query = self.scaler.transform(features)
        selected = {
            name
            for name, _ in sorted(query.items(), key=lambda item: (-abs(item[1]), item[0]))[
                : self.max_features
            ]
        }
        if not selected:
            return 0.5

        # Standardising every stored neighbour from scratch on every prediction is
        # the whole cost of this model: 160 neighbours x 48 features x a square
        # root each, asked again for each of the three horizons, on each bar. The
        # statistics do not depend on the neighbour, so they come out of the loop —
        # and only the selected features are standardised at all, which is the
        # difference between asking about 48 numbers and constructing 150.
        scaler = self.scaler
        stats = {name: scaler.stats(name) for name in selected}
        query = {name: query[name] for name in selected}

        distances: list[tuple[float, int, int]] = []
        for age, (raw, target) in enumerate(reversed(self.samples)):
            squared = 0.0
            for name, (mean, scale) in stats.items():
                value = raw.get(name)
                neighbour = 0.0 if value is None else scaler.standardise(value, mean, scale)
                difference = query[name] - neighbour
                squared += difference * difference
            distances.append((math.sqrt(squared / len(stats)), age, target))

        nearest = sorted(distances, key=lambda item: (item[0], item[1]))[: min(self.k, len(distances))]

        weighted_up = 0.5 * self.prior_strength
        total = self.prior_strength
        for distance, age, target in nearest:
            recency = math.exp(-math.log(2.0) * age / self.half_life)
            weight = recency / max(distance, 0.05)
            weighted_up += weight * target
            total += weight
        return weighted_up / total if total else 0.5

    def update(self, features: dict[str, float], target: int) -> None:
        # Keep the state checkpoint bounded in both rows and columns. Selection
        # uses only the causal scaler and the current feature vector (never the
        # target), so this compression cannot leak outcome information.
        scaled = self.scaler.transform(features)
        names = [
            name
            for name, _ in sorted(scaled.items(), key=lambda item: (-abs(item[1]), item[0]))[
                : self.max_features
            ]
        ]
        self.samples.append(({name: features[name] for name in names}, int(bool(target))))
        if len(self.samples) > self.window:
            self.samples = self.samples[-self.window :]
        self.samples_seen += 1
        self.scaler.update(features)

    def state_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "k": self.k,
            "window": self.window,
            "half_life": self.half_life,
            "max_features": self.max_features,
            "prior_strength": self.prior_strength,
            "samples_seen": self.samples_seen,
            "samples": [{"features": features, "target": target} for features, target in self.samples],
            "scaler": self.scaler.as_dict(),
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.k = int(payload.get("k", self.k))
        self.window = int(payload.get("window", self.window))
        self.half_life = float(payload.get("half_life", self.half_life))
        self.max_features = int(payload.get("max_features", self.max_features))
        self.prior_strength = float(payload.get("prior_strength", self.prior_strength))
        self.samples_seen = int(payload.get("samples_seen", 0))
        self.samples = [
            (
                {str(name): float(value) for name, value in dict(item.get("features", {})).items()},
                int(bool(item.get("target", 0))),
            )
            for item in list(payload.get("samples", []))[-self.window :]
        ]
        self.scaler = CausalScaler.from_dict(dict(payload.get("scaler", {})))


__all__ = ["OnlineAdaptiveKNN"]
