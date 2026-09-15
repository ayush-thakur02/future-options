"""The money simulator as a plugin.

Provides the ``money_simulator`` capability: a funded paper book that trades the
index, the call and the put against the live tape and keeps score in rupees.

It is an *advisory* pack rather than a forecast because that is what it is — it
does not have an opinion about the market that the platform does not already
have. It consumes the strategy ensemble, the online learners, the model and the
projected path, and it answers a different question about them: what happens to
money if you act on this, at this size, paying these costs.

Because it is a plugin, replacing it is a matter of supplying another pack that
provides the same capability. A different sizing rule, a spread-selling book, or
a five-leg version needs no change to the board or the runtime.
"""

from __future__ import annotations

from pathlib import Path

from core.settings import Settings
from kernel import PluginContext, PluginKind, PluginManifest

from .ledger import SimulationLedger
from .policy import PolicyWeights
from .simulator import MoneySimulator, SimulatorConfig

MANIFEST = PluginManifest(
    name="money_simulator",
    kind=PluginKind.ADVISORY,
    description="A funded paper book trading the index, call and put, scored in rupees",
    provides=("money_simulator",),
    tags=("simulation", "paper-trading", "risk", "rupees", "persistent"),
    params={
        "opening_balance": 25_000.0,
        "reserve": 200_000.0,
        "lot_size": 65,
        "decision_seconds": 10.0,
        "state_path": None,
    },
)


def config_from(params: dict) -> SimulatorConfig:
    """Build the simulator's settings from a plugin YAML mapping.

    Unknown keys are ignored rather than rejected so that a stale config file
    cannot stop the dashboard from starting, and so a newer YAML can be dropped
    onto an older build without a migration step.
    """
    weights = params.get("weights")
    return SimulatorConfig(
        opening_balance=float(params.get("opening_balance", 25_000.0)),
        reserve=float(params.get("reserve", 200_000.0)),
        lot_size=int(params.get("lot_size", 65)),
        decision_seconds=float(params.get("decision_seconds", 10.0)),
        entry_timeout_seconds=float(params.get("entry_timeout_seconds", 30.0)),
        horizon_bars=int(params.get("horizon_bars", 3)),
        entry_view=float(params.get("entry_view", 0.15)),
        min_edge_bps=float(params.get("min_edge_bps", 0.0)),
        min_confirmations=int(params.get("min_confirmations", 2)),
        require_projection=bool(params.get("require_projection", True)),
        min_equity_fraction=float(params.get("min_equity_fraction", 0.5)),
        stop_atr=float(params.get("stop_atr", 1.5)),
        target_atr=float(params.get("target_atr", 2.5)),
        fallback_stop_bps=float(params.get("fallback_stop_bps", 10.0)),
        max_hold_minutes=float(params.get("max_hold_minutes", 5.0)),
        min_hold_seconds=float(params.get("min_hold_seconds", 45.0)),
        cooldown_seconds=float(params.get("cooldown_seconds", 45.0)),
        reversal_view=float(params.get("reversal_view", -0.15)),
        option_cost_rate=float(params.get("option_cost_rate", 0.012)),
        projection_scale_bps=float(params.get("projection_scale_bps", 4.0)),
        weights=PolicyWeights(
            strategy=float((weights or {}).get("strategy", 1.0)),
            ai=float((weights or {}).get("ai", 0.8)),
            model=float((weights or {}).get("model", 0.5)),
            projection=float((weights or {}).get("projection", 1.5)),
            momentum=float((weights or {}).get("momentum", 0.0)),
        ),
    )


def state_path(settings: Settings, params: dict, mode: str) -> Path:
    configured = params.get("state_path")
    if configured:
        return Path(str(configured))
    return settings.data_dir / "research" / "money_simulator" / f"{mode}.sqlite3"


def build(
    ctx: PluginContext,
    mode: str = "simulation",
    state_path: str | Path | None = None,
    **params,
) -> MoneySimulator:
    """Build the book, with the platform's own cost model for the index leg."""
    from backtest.costs import CostModel

    settings = ctx.settings
    config = config_from(params)
    selected = Path(state_path) if state_path else None
    ledger = SimulationLedger(
        selected or settings.data_dir / "research" / "money_simulator" / f"{mode}.sqlite3",
        mode=mode,
    )
    return MoneySimulator(
        config,
        ledger=ledger,
        cost_model=CostModel(slippage_bps=settings.slippage_bps, lot_size=config.lot_size),
    )


__all__ = ["MANIFEST", "MoneySimulator", "SimulatorConfig", "build", "config_from", "state_path"]
