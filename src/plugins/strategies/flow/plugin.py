"""Price-volume flow strategies as an independently discoverable plugin."""

from __future__ import annotations

from kernel import PluginContext, PluginKind, PluginManifest

from ..base import StrategyPack
from .rules import STRATEGIES

MANIFEST = PluginManifest(
    name="flow",
    kind=PluginKind.STRATEGY,
    version="0.2.0",
    description="Three participation rules: Chaikin confirmation, MFI reversal, and OBV divergence",
    provides=tuple(f"strategy:{cls.name}" for cls in STRATEGIES),
    tags=("rules", "volume", "flow", "participation", "causal"),
)


def build(ctx: PluginContext, **params) -> StrategyPack:
    return StrategyPack(
        name="flow",
        category="flow",
        description=MANIFEST.description,
        instances=tuple(cls() for cls in STRATEGIES),
    )
