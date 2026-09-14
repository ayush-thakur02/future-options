"""Channel-structure strategies as an independently discoverable plugin."""

from __future__ import annotations

from kernel import PluginContext, PluginKind, PluginManifest

from ..base import StrategyPack, make_strategy_pack
from .rules import STRATEGIES

MANIFEST = PluginManifest(
    name="channels",
    kind=PluginKind.STRATEGY,
    version="0.2.0",
    description=(
        "Three channel rules: Keltner continuation, regression-channel expansion, "
        "and sustained auction pressure"
    ),
    provides=tuple(f"strategy:{cls.name}" for cls in STRATEGIES),
    tags=("rules", "channels", "breakout", "causal"),
)


def build(ctx: PluginContext, parameters=None, weights=None, **params) -> StrategyPack:
    return make_strategy_pack(
        name="channels",
        category="channels",
        description=MANIFEST.description,
        strategy_types=STRATEGIES,
        parameters=parameters,
        weights=weights,
    )
