"""Adaptive regime strategies as an independently discoverable plugin."""

from __future__ import annotations

from kernel import PluginContext, PluginKind, PluginManifest

from ..base import StrategyPack, make_strategy_pack
from .rules import STRATEGIES

MANIFEST = PluginManifest(
    name="regime",
    kind=PluginKind.STRATEGY,
    version="0.2.0",
    description="Three adaptive rules: Hurst routing, directional entropy, and volatility-state rotation",
    provides=tuple(f"strategy:{cls.name}" for cls in STRATEGIES),
    tags=("rules", "regime", "adaptive", "entropy", "causal"),
)


def build(ctx: PluginContext, parameters=None, weights=None, **params) -> StrategyPack:
    return make_strategy_pack(
        name="regime",
        category="regime",
        description=MANIFEST.description,
        strategy_types=STRATEGIES,
        parameters=parameters,
        weights=weights,
    )
