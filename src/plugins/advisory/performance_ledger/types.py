"""Public strategy outcome and scorecard records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class StrategyOutcome:
    strategy: str
    instrument: str
    target_at: datetime
    direction: str
    hit: bool
    gross_pnl_bps: float
    net_pnl_bps: float


@dataclass(frozen=True, slots=True)
class StrategyScorecard:
    strategy: str
    instrument: str
    samples: int
    hits: int
    accuracy: float
    wins: int
    losses: int
    gross_pnl_bps: float
    cost_paid_bps: float
    profit_bps: float
    loss_bps: float
    net_pnl_bps: float
    mean_net_pnl_bps: float
    max_drawdown_bps: float
    trust_score: float

    def as_row(self) -> dict:
        return {
            "strategy": self.strategy,
            "instrument": self.instrument,
            "samples": self.samples,
            "accuracy": round(self.accuracy, 4),
            "wins": self.wins,
            "losses": self.losses,
            "net_pnl_bps": round(self.net_pnl_bps, 2),
            "drawdown_bps": round(self.max_drawdown_bps, 2),
            "trust_score": round(self.trust_score, 4),
        }


__all__ = ["StrategyOutcome", "StrategyScorecard"]
