"""Five momentum rules as a plugin pack.

Each class in ``rules.py`` declares a name and a category; the manifest below
derives one ``strategy:<name>`` capability per class, so the kernel can report
what this build offers without anything importing the pack.
"""

from __future__ import annotations

from kernel import PluginContext, PluginKind, PluginManifest

from ..base import StrategyPack
from .rules import STRATEGIES

MANIFEST = PluginManifest(
    name="momentum",
    kind=PluginKind.STRATEGY,
    description="Five momentum rules: MACD, rate of change, stochastic, activity spike, order flow",
    provides=tuple(f"strategy:{cls.name}" for cls in STRATEGIES),
    tags=("rules", "momentum"),
)


def build(ctx: PluginContext, **params) -> StrategyPack:
    """Instantiate every rule in the pack. Strategies are stateless, so this is cheap."""
    return StrategyPack(
        name="momentum",
        category="momentum",
        description=MANIFEST.description,
        instances=tuple(cls() for cls in STRATEGIES),
    )
