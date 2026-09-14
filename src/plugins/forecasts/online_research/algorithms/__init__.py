"""Registry of independently replaceable online learners."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from .base import OnlineClassifier
from .gaussian_nb import OnlineGaussianNB
from .logistic import OnlineLogistic
from .passive_aggressive import OnlinePassiveAggressive

AlgorithmFactory = Callable[[], OnlineClassifier]

ALGORITHMS: dict[str, AlgorithmFactory] = {
    OnlineLogistic.name: OnlineLogistic,
    OnlinePassiveAggressive.name: OnlinePassiveAggressive,
    OnlineGaussianNB.name: OnlineGaussianNB,
}


def build_algorithms(names: Iterable[str] | None = None) -> dict[str, OnlineClassifier]:
    selected = tuple(names) if names is not None else tuple(ALGORITHMS)
    if not selected:
        raise ValueError("at least one online algorithm is required")
    unknown = sorted(set(selected) - set(ALGORITHMS))
    if unknown:
        raise ValueError(
            f"unknown online algorithms: {', '.join(unknown)}; available: {', '.join(ALGORITHMS)}"
        )
    return {name: ALGORITHMS[name]() for name in selected}


__all__ = [
    "ALGORITHMS",
    "AlgorithmFactory",
    "OnlineClassifier",
    "OnlineGaussianNB",
    "OnlineLogistic",
    "OnlinePassiveAggressive",
    "build_algorithms",
]
