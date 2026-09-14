"""Kernel entry point for causal, restart-safe online research scoring."""

from __future__ import annotations

from pathlib import Path

from kernel import PluginContext, PluginKind, PluginManifest

from .engine import OnlineResearchLab

MANIFEST = PluginManifest(
    name="online_research",
    kind=PluginKind.FORECAST,
    description="Causal incremental learners with persistent accuracy, calibration, and P&L research",
    provides=("online_research",),
    tags=("online-learning", "prequential", "research", "persistent"),
    params={
        "rolling_window": 100,
        "min_trust_samples": 50,
        "min_trust_for_action": 0.15,
        "buy_probability": 0.56,
        "sell_probability": 0.44,
    },
)


def build(
    ctx: PluginContext,
    state_path: str | Path | None = None,
    algorithms=None,
    rolling_window: int = 100,
    min_trust_samples: int = 50,
    min_trust_for_action: float = 0.15,
    buy_probability: float = 0.56,
    sell_probability: float = 0.44,
    cost_bps: float | None = None,
    **params,
) -> OnlineResearchLab:
    path = Path(state_path) if state_path else ctx.settings.data_dir / "research" / "online_ai.sqlite3"
    if isinstance(algorithms, dict):
        selected = {
            str(name): dict(values or {}) for name, values in algorithms.items()
        }
    elif isinstance(algorithms, str):
        selected = (algorithms,)
    else:
        selected = tuple(str(name) for name in algorithms) if algorithms else None
    default_cost = ctx.settings.cost_hurdle_bps() if cost_bps is None else float(cost_bps)
    return OnlineResearchLab(
        path,
        algorithms=selected,
        rolling_window=rolling_window,
        min_trust_samples=min_trust_samples,
        min_trust_for_action=min_trust_for_action,
        buy_probability=buy_probability,
        sell_probability=sell_probability,
        default_cost_bps=default_cost,
    )


__all__ = ["MANIFEST", "build"]
