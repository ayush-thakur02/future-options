"""Five trend-following rules as a plugin pack.

Each class in ``rules.py`` declares a name and a category; the manifest below
derives one ``strategy:<name>`` capability per class, so the kernel can report
what this build offers without anything importing the pack.
"""

from __future__ import annotations

from kernel import PluginContext, PluginKind, PluginManifest

from ..base import StrategyPack
from .rules import STRATEGIES

MANIFEST = PluginManifest(
    name="trend",
    kind=PluginKind.STRATEGY,
    description="Five trend-following rules: EMA, supertrend, Donchian, opening range, efficiency",
    provides=tuple(f"strategy:{cls.name}" for cls in STRATEGIES),
    tags=("rules", "trend"),
)


def build(ctx: PluginContext, **params) -> StrategyPack:
    """Instantiate every rule in the pack. Strategies are stateless, so this is cheap."""
    return StrategyPack(
        name="trend",
        category="trend",
        description=MANIFEST.description,
        instances=tuple(cls() for cls in STRATEGIES),
    )
