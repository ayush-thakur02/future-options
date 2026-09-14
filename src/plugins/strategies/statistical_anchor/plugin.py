"""Cross-instrument strategies as an independently discoverable plugin."""

from __future__ import annotations

from kernel import PluginContext, PluginKind, PluginManifest

from ..base import StrategyPack
from .rules import STRATEGIES

MANIFEST = PluginManifest(
    name="statistical_anchor",
    kind=PluginKind.STRATEGY,
    version="0.2.0",
    description=(
        "Two beta-aware cross-instrument rules: spread reversion and relative-strength rotation; "
        "neutral until an anchor series is supplied"
    ),
    provides=tuple(f"strategy:{cls.name}" for cls in STRATEGIES),
    tags=("rules", "statistical", "relative-value", "anchor", "causal"),
)


def build(ctx: PluginContext, **params) -> StrategyPack:
    return StrategyPack(
        name="statistical_anchor",
        category="statistical",
        description=MANIFEST.description,
        instances=tuple(cls() for cls in STRATEGIES),
    )
