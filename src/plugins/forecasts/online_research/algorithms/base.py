"""Small contracts and causal feature scaling for online classifiers."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


def sigmoid(value: float) -> float:
    """A numerically stable sigmoid."""
    if value >= 0:
        inverse = math.exp(-min(value, 60.0))
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(max(value, -60.0))
    return exponent / (1.0 + exponent)


@dataclass(slots=True)
class RunningMoment:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    @property
    def standard_deviation(self) -> float:
        if self.count < 2:
            return 0.0
        return math.sqrt(max(self.m2 / (self.count - 1), 0.0))

    def as_dict(self) -> dict[str, float | int]:
        return {"count": self.count, "mean": self.mean, "m2": self.m2}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunningMoment:
        return cls(
            count=int(payload.get("count", 0)),
            mean=float(payload.get("mean", 0.0)),
            m2=float(payload.get("m2", 0.0)),
        )


class CausalScaler:
    """Standardise against matured samples only.

    ``transform`` never mutates state. The caller transforms and trains first,
    then records the observation, so even the feature distribution used for an
    update contains no information from that sample or its outcome.
    """

    def __init__(self, clip: float = 8.0) -> None:
        self.clip = float(clip)
        self.moments: dict[str, RunningMoment] = {}

    def transform(self, features: dict[str, float]) -> dict[str, float]:
        transformed: dict[str, float] = {}
        for name, value in features.items():
            moment = self.moments.get(name)
            if moment is None:
                transformed[name] = 0.0
                continue
            scale = moment.standard_deviation
            normalised = (value - moment.mean) / scale if scale > 1e-12 else value - moment.mean
            transformed[name] = max(-self.clip, min(self.clip, normalised))
        return transformed

    def update(self, features: dict[str, float]) -> None:
        for name, value in features.items():
            self.moments.setdefault(name, RunningMoment()).update(value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "clip": self.clip,
            "moments": {name: moment.as_dict() for name, moment in self.moments.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CausalScaler:
        scaler = cls(clip=float(payload.get("clip", 8.0)))
        scaler.moments = {
            str(name): RunningMoment.from_dict(dict(moment))
            for name, moment in dict(payload.get("moments", {})).items()
        }
        return scaler


class OnlineClassifier(ABC):
    """Probability classifier trained one matured outcome at a time."""

    name: str

    @abstractmethod
    def predict_proba(self, features: dict[str, float]) -> float:
        """Return P(up) without changing model state."""

    @abstractmethod
    def update(self, features: dict[str, float], target: int) -> None:
        """Learn from a target that is now observable."""

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return JSON-safe state."""

    @abstractmethod
    def load_state_dict(self, payload: dict[str, Any]) -> None:
        """Restore state from ``state_dict``."""


class ScaledLinearClassifier(OnlineClassifier):
    """Shared persistence and feature handling for sparse linear learners."""

    def __init__(self) -> None:
        self.weights: dict[str, float] = {}
        self.bias = 0.0
        self.samples_seen = 0
        self.scaler = CausalScaler()

    def margin(self, features: dict[str, float]) -> float:
        scaled = self.scaler.transform(features)
        return self.bias + sum(self.weights.get(name, 0.0) * value for name, value in scaled.items())

    def state_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "weights": dict(self.weights),
            "bias": self.bias,
            "samples_seen": self.samples_seen,
            "scaler": self.scaler.as_dict(),
            **self._parameters(),
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.weights = {
            str(name): float(value) for name, value in dict(payload.get("weights", {})).items()
        }
        self.bias = float(payload.get("bias", 0.0))
        self.samples_seen = int(payload.get("samples_seen", 0))
        self.scaler = CausalScaler.from_dict(dict(payload.get("scaler", {})))
        self._load_parameters(payload)

    def _parameters(self) -> dict[str, Any]:
        return {}

    def _load_parameters(self, payload: dict[str, Any]) -> None:
        return None


__all__ = [
    "CausalScaler",
    "OnlineClassifier",
    "RunningMoment",
    "ScaledLinearClassifier",
    "sigmoid",
]
