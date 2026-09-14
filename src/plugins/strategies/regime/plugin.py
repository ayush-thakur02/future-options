"""Adaptive regime strategies as an independently discoverable plugin."""

from __future__ import annotations

from kernel import PluginContext, PluginKind, PluginManifest

from ..base import StrategyPack
from .rules import STRATEGIES

MANIFEST = PluginManifest(
    name="regime",
    kind=PluginKind.STRATEGY,
    version="0.2.0",
    description="Three adaptive rules: Hurst routing, directional entropy, and volatility-state rotation",
    provides=tuple(f"strategy:{cls.name}" for cls in STRATEGIES),
    tags=("rules", "regime", "adaptive", "entropy", "causal"),
)


def build(ctx: PluginContext, **params) -> StrategyPack:
    return StrategyPack(
        name="regime",
        category="regime",
        description=MANIFEST.description,
        instances=tuple(cls() for cls in STRATEGIES),
    )
