"""Persistent scorecards for every active strategy plugin."""

from __future__ import annotations

from pathlib import Path

from kernel import PluginContext, PluginKind, PluginManifest

from .ledger import StrategyPerformanceLedger

MANIFEST = PluginManifest(
    name="performance_ledger",
    kind=PluginKind.ADVISORY,
    description="Restart-safe strategy accuracy, hypothetical P&L, drawdown and trust",
    provides=("strategy_performance",),
    tags=("research", "strategy", "scorecard", "persistent"),
    params={"min_trust_samples": 50},
)


def build(
    ctx: PluginContext,
    path: str | Path | None = None,
    min_trust_samples: int = 50,
    **params,
) -> StrategyPerformanceLedger:
    selected = Path(path) if path else ctx.settings.data_dir / "research" / "strategies.sqlite3"
    return StrategyPerformanceLedger(selected, min_trust_samples=min_trust_samples)


__all__ = ["MANIFEST", "build"]
