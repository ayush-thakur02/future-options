"""Performance statistics for a completed backtest.

Reported alongside gross numbers so the effect of costs is always visible. A
strategy with a positive gross expectancy and a negative net one is extremely
common on intraday index data, and a report that only shows net PnL hides why.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..models import Trade

TRADING_DAYS_PER_YEAR = 250


@dataclass
class PerformanceReport:
    trades: int
    wins: int
    losses: int
    win_rate: float
    gross_pnl: float
    net_pnl: float
    total_costs: float
    avg_gross_bps: float
    avg_net_bps: float
    avg_win_bps: float
    avg_loss_bps: float
    profit_factor: float
    expectancy_bps: float
    max_drawdown_bps: float
    sharpe: float
    sortino: float
    bars_in_market: int
    total_bars: int
    long_trades: int = 0
    short_trades: int = 0
    long_win_rate: float = 0.0
    short_win_rate: float = 0.0
    cost_drag_bps: float = 0.0
    avg_notional: float = 0.0
    extras: dict = field(default_factory=dict)

    @property
    def exposure(self) -> float:
        return self.bars_in_market / self.total_bars if self.total_bars else 0.0

    def as_row(self) -> dict:
        return {
            "trades": self.trades,
            "win_rate": round(self.win_rate, 4),
            "gross_bps": round(self.avg_gross_bps, 3),
            "net_bps": round(self.avg_net_bps, 3),
            "costs_bps": round(self.cost_drag_bps, 3),
            "profit_factor": round(self.profit_factor, 3),
            "expectancy_bps": round(self.expectancy_bps, 3),
            "sharpe": round(self.sharpe, 3),
            "max_dd_bps": round(self.max_drawdown_bps, 1),
            "exposure": round(self.exposure, 4),
        }

    def lines(self) -> list[str]:
        return [
            f"trades               {self.trades:,} ({self.long_trades} long / {self.short_trades} short)",
            f"avg position         Rs {self.avg_notional:,.0f}",
            f"win rate             {self.win_rate:.2%}",
            f"gross per trade      {self.avg_gross_bps:+.2f} bps",
            f"costs per trade      {self.cost_drag_bps:.2f} bps",
            f"net per trade        {self.avg_net_bps:+.2f} bps",
            f"expectancy           {self.expectancy_bps:+.2f} bps",
            f"profit factor        {self.profit_factor:.3f}",
            f"avg win / avg loss   {self.avg_win_bps:+.2f} / {self.avg_loss_bps:+.2f} bps",
            f"net PnL              Rs {self.net_pnl:+,.0f}  (costs Rs {self.total_costs:,.0f})",
            f"Sharpe (annualised)  {self.sharpe:+.3f}",
            f"Sortino (annualised) {self.sortino:+.3f}",
            f"max drawdown         {self.max_drawdown_bps:.1f} bps",
            f"time in market       {self.exposure:.1%}",
        ]


def build_report(
    trades: list[Trade],
    total_bars: int,
    bars_in_market: int,
    notional: float,
) -> PerformanceReport:
    if not trades:
        return PerformanceReport(
            trades=0, wins=0, losses=0, win_rate=0.0, gross_pnl=0.0, net_pnl=0.0,
            total_costs=0.0, avg_gross_bps=0.0, avg_net_bps=0.0, avg_win_bps=0.0,
            avg_loss_bps=0.0, profit_factor=0.0, expectancy_bps=0.0,
            max_drawdown_bps=0.0, sharpe=0.0, sortino=0.0,
            bars_in_market=0, total_bars=total_bars,
        )

    entry_prices = np.array([trade.entry_price for trade in trades], dtype="float64")
    signs = np.array([trade.direction.sign for trade in trades], dtype="float64")
    exit_prices = np.array([trade.exit_price for trade in trades], dtype="float64")
    costs = np.array([trade.costs for trade in trades], dtype="float64")
    quantities = np.array([trade.quantity for trade in trades], dtype="float64")

    # Rupee exposure per trade. Costs and PnL are both expressed against this,
    # not against the price level — dividing a cost by the index price rather
    # than by the position value overstates it by the position's size.
    notionals = np.maximum(entry_prices * quantities, 1e-9)

    gross_bps = (exit_prices / entry_prices - 1.0) * signs * 10_000.0
    costs_bps = costs / notionals * 10_000.0
    net_bps = gross_bps - costs_bps

    gross_pnl_per_trade = gross_bps / 10_000.0 * notionals
    net_pnl_per_trade = net_bps / 10_000.0 * notionals

    wins = net_bps > 0
    win_count = int(wins.sum())
    loss_count = int((~wins).sum())

    gains = net_bps[wins].sum()
    losses = -net_bps[~wins].sum()
    profit_factor = float(gains / losses) if losses > 0 else float("inf") if gains > 0 else 0.0

    equity = np.cumsum(net_bps)
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak

    sharpe = _annualised_sharpe(net_bps, trades)
    sortino = _annualised_sortino(net_bps, trades)

    long_mask = signs > 0
    short_mask = ~long_mask

    return PerformanceReport(
        trades=len(trades),
        wins=win_count,
        losses=loss_count,
        win_rate=win_count / len(trades),
        gross_pnl=float(gross_pnl_per_trade.sum()),
        net_pnl=float(net_pnl_per_trade.sum()),
        total_costs=float(costs.sum()),
        avg_gross_bps=float(gross_bps.mean()),
        avg_net_bps=float(net_bps.mean()),
        avg_win_bps=float(net_bps[wins].mean()) if win_count else 0.0,
        avg_loss_bps=float(net_bps[~wins].mean()) if loss_count else 0.0,
        profit_factor=profit_factor,
        expectancy_bps=float(net_bps.mean()),
        max_drawdown_bps=float(drawdown.min()) if len(drawdown) else 0.0,
        sharpe=sharpe,
        sortino=sortino,
        bars_in_market=bars_in_market,
        total_bars=total_bars,
        long_trades=int(long_mask.sum()),
        short_trades=int(short_mask.sum()),
        long_win_rate=float(wins[long_mask].mean()) if long_mask.any() else 0.0,
        short_win_rate=float(wins[short_mask].mean()) if short_mask.any() else 0.0,
        cost_drag_bps=float(costs_bps.mean()),
        avg_notional=float(notionals.mean()),
    )


def _annualised_sharpe(net_bps: np.ndarray, trades: list[Trade]) -> float:
    if len(net_bps) < 2:
        return 0.0
    std = net_bps.std(ddof=1)
    if std <= 0:
        return 0.0
    trades_per_year = _trades_per_year(trades)
    return float(net_bps.mean() / std * np.sqrt(trades_per_year))


def _annualised_sortino(net_bps: np.ndarray, trades: list[Trade]) -> float:
    if len(net_bps) < 2:
        return 0.0
    downside = net_bps[net_bps < 0]
    if len(downside) == 0:
        return float("inf") if net_bps.mean() > 0 else 0.0
    downside_dev = np.sqrt((downside**2).mean())
    if downside_dev <= 0:
        return 0.0
    return float(net_bps.mean() / downside_dev * np.sqrt(_trades_per_year(trades)))


def _trades_per_year(trades: list[Trade]) -> float:
    """Trade frequency, used to annualise the ratio statistics."""
    if len(trades) < 2:
        return TRADING_DAYS_PER_YEAR
    first = trades[0].entry_ts
    last = trades[-1].exit_ts
    span_days = max((last - first).total_seconds() / 86_400.0, 1.0)
    return (len(trades) / span_days) * 365.0


def equity_curve(trades: list[Trade]) -> pd.Series:
    if not trades:
        return pd.Series(dtype="float64")
    frame = pd.DataFrame(
        {
            "ts": [trade.exit_ts for trade in trades],
            "ret": [trade.return_bps for trade in trades],
        }
    ).set_index("ts")
    return frame["ret"].cumsum()
