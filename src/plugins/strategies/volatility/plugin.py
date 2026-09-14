"""Four volatility rules as a plugin pack.

Each class in ``rules.py`` declares a name and a category; the manifest below
derives one ``strategy:<name>`` capability per class, so the kernel can report
what this build offers without anything importing the pack.
"""

from __future__ import annotations

from kernel import PluginContext, PluginKind, PluginManifest

from ..base import StrategyPack, make_strategy_pack
from .rules import STRATEGIES

MANIFEST = PluginManifest(
    name="volatility",
    kind=PluginKind.STRATEGY,
    description="Four volatility rules: squeeze release, breakout, vol regime, GARCH forecast",
    provides=tuple(f"strategy:{cls.name}" for cls in STRATEGIES),
    tags=("rules", "volatility"),
)


def build(ctx: PluginContext, parameters=None, weights=None, **params) -> StrategyPack:
    """Instantiate every rule in the pack. Strategies are stateless, so this is cheap."""
    return make_strategy_pack(
        name="volatility",
        category="volatility",
        description=MANIFEST.description,
        strategy_types=STRATEGIES,
        parameters=parameters,
        weights=weights,
    )
