"""Registry of independently replaceable online learners."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

from .adaptive_knn import OnlineAdaptiveKNN
from .base import OnlineClassifier
from .ftrl import OnlineFTRL
from .gaussian_nb import OnlineGaussianNB
from .logistic import OnlineLogistic
from .passive_aggressive import OnlinePassiveAggressive
from .strategy_combinations import OnlineStrategyCombinations

AlgorithmFactory = Callable[..., OnlineClassifier]

ALGORITHMS: dict[str, AlgorithmFactory] = {
    OnlineLogistic.name: OnlineLogistic,
    OnlinePassiveAggressive.name: OnlinePassiveAggressive,
    OnlineGaussianNB.name: OnlineGaussianNB,
    OnlineFTRL.name: OnlineFTRL,
    OnlineAdaptiveKNN.name: OnlineAdaptiveKNN,
    OnlineStrategyCombinations.name: OnlineStrategyCombinations,
}


def build_algorithms(
    names: Iterable[str] | Mapping[str, Mapping] | None = None,
) -> dict[str, OnlineClassifier]:
    parameters: dict[str, dict] = {}
    if isinstance(names, Mapping):
        selected = tuple(str(name) for name in names)
        parameters = {str(name): dict(values or {}) for name, values in names.items()}
    else:
        selected = tuple(names) if names is not None else tuple(ALGORITHMS)
    if not selected:
        raise ValueError("at least one online algorithm is required")
    unknown = sorted(set(selected) - set(ALGORITHMS))
    if unknown:
        raise ValueError(
            f"unknown online algorithms: {', '.join(unknown)}; available: {', '.join(ALGORITHMS)}"
        )
    return {name: ALGORITHMS[name](**parameters.get(name, {})) for name in selected}


__all__ = [
    "ALGORITHMS",
    "AlgorithmFactory",
    "OnlineAdaptiveKNN",
    "OnlineClassifier",
    "OnlineFTRL",
    "OnlineGaussianNB",
    "OnlineLogistic",
    "OnlinePassiveAggressive",
    "OnlineStrategyCombinations",
    "build_algorithms",
]
