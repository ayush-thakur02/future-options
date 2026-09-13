"""Event-driven backtest over a vectorised signal series.

Execution timing is the detail that matters most here, and it is where most
backtests quietly cheat. A signal computed from bar *t*'s close cannot be filled
at bar *t*'s close — by the time you know it, that price is gone. So the engine
always enters at the **next bar's open**, which is the first price actually
available to a decision made at the close.

The rest is deliberately conservative:

* One position at a time. Overlapping trades are how a strategy appears to have
  far more edge than it does, because the same move gets counted repeatedly.
* Trades that would run past the session close are skipped rather than carried
  overnight, since the platform forecasts intraday moves only.
* Costs are charged on both legs of every trade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from ..models import Direction, Trade
from ..trading_calendar import IST
from .costs import CostModel
from .report import PerformanceReport, build_report


@dataclass
class BacktestConfig:
    """Parameters controlling execution and sizing."""

    horizon: int = 5
    entry_threshold: float = 0.15
    # Target exposure. Positions size in whole lots, so the realised exposure is
    # the nearest whole lot to this — and at least one lot, which for NIFTY near
    # 24,000 with a 75-unit contract is roughly Rs 1.8 million.
    notional: float = 2_000_000.0
    lot_size: int = 75
    allow_long: bool = True
    allow_short: bool = True
    cost_model: CostModel = field(default_factory=CostModel)
    require_same_session: bool = True

    def size_for(self, price: float) -> tuple[int, float]:
        """Lots and units for a position near ``price``.

        Positions are sized in whole lots, which is how index futures actually
        trade. ``notional`` is the rupee exposure being targeted; the realised
        exposure is the nearest whole number of lots to it. Treating the target
        notional as a unit count instead would multiply the position by the price
        level and inflate both PnL and costs by four orders of magnitude.
        """
        lot_size = max(self.lot_size, 1)
        if price <= 0:
            return 1, lot_size
        lots = max(int(round(self.notional / (price * lot_size))), 1)
        return lots, lots * lot_size


@dataclass
class BacktestResult:
    """Trades, equity, and statistics from one run."""

    trades: list[Trade]
    report: PerformanceReport
    config: BacktestConfig
    equity: pd.Series
    scores: pd.Series
    strategy: str = ""

    def summary(self) -> dict:
        row = self.report.as_row()
        row["strategy"] = self.strategy
        row["horizon"] = self.config.horizon
        return row


def run_backtest(
    bars: pd.DataFrame,
    scores: pd.Series,
    config: BacktestConfig,
    strategy: str = "",
) -> BacktestResult:
    """Simulate trading the given conviction series over ``bars``."""
    if bars.empty or scores.empty:
        empty = build_report([], len(bars), 0, config.notional)
        return BacktestResult([], empty, config, pd.Series(dtype="float64"), scores, strategy)

    frame = bars[["open", "high", "low", "close"]].copy()
    alerts = scores.reindex(frame.index).fillna(0.0).to_numpy(dtype="float64")
    opens = frame["open"].to_numpy(dtype="float64")
    closes = frame["close"].to_numpy(dtype="float64")
    times = frame.index
    day = pd.Series(times.normalize(), index=frame.index).to_numpy()
    n = len(frame)

    trades: list[Trade] = []
    bars_in_market = 0
    next_available = 0

    for i in range(n - 1):
        if i < next_available:
            continue

        score = alerts[i]
        if abs(score) < config.entry_threshold:
            continue

        direction = Direction.UP if score > 0 else Direction.DOWN
        if direction is Direction.UP and not config.allow_long:
            continue
        if direction is Direction.DOWN and not config.allow_short:
            continue

        entry_idx = i + 1
        exit_idx = entry_idx + config.horizon
        if exit_idx >= n:
            break

        # Never hold across the session boundary.
        if config.require_same_session and day[exit_idx] != day[entry_idx]:
            continue

        entry_price = opens[entry_idx]
        exit_price = closes[exit_idx]
        if not np.isfinite(entry_price) or not np.isfinite(exit_price) or entry_price <= 0:
            continue

        _, units = config.size_for(entry_price)
        position_notional = entry_price * units
        costs = config.cost_model.round_trip_cost(position_notional)

        trades.append(
            Trade(
                entry_ts=_as_datetime(times[entry_idx]),
                exit_ts=_as_datetime(times[exit_idx]),
                direction=direction,
                entry_price=float(entry_price),
                exit_price=float(exit_price),
                quantity=units,
                costs=float(costs),
                strategy=strategy,
            )
        )
        bars_in_market += config.horizon
        next_available = exit_idx + 1

    report = build_report(trades, total_bars=n, bars_in_market=bars_in_market, notional=config.notional)
    equity = _equity_from_trades(trades)
    return BacktestResult(trades, report, config, equity, scores, strategy)


def _equity_from_trades(trades: list[Trade]) -> pd.Series:
    if not trades:
        return pd.Series(dtype="float64")
    frame = pd.DataFrame(
        {
            "ts": [trade.exit_ts for trade in trades],
            "ret": [trade.return_bps for trade in trades],
        }
    ).set_index("ts")
    frame.index = pd.DatetimeIndex(frame.index)
    return frame["ret"].cumsum()


def _as_datetime(stamp) -> datetime:
    return stamp.to_pydatetime() if hasattr(stamp, "to_pydatetime") else stamp


def run_threshold_sweep(
    bars: pd.DataFrame,
    scores: pd.Series,
    config: BacktestConfig,
    thresholds: tuple[float, ...] = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5),
    strategy: str = "",
) -> pd.DataFrame:
    """Backtest across entry thresholds.

    Raising the threshold trades less but more selectively. If net expectancy does
    not improve as the threshold rises, the signal carries no information about
    its own confidence — which is a more useful finding than any single run.
    """
    rows: list[dict] = []
    for threshold in thresholds:
        tuned = BacktestConfig(
            horizon=config.horizon,
            entry_threshold=threshold,
            notional=config.notional,
            lot_size=config.lot_size,
            allow_long=config.allow_long,
            allow_short=config.allow_short,
            cost_model=config.cost_model,
            require_same_session=config.require_same_session,
        )
        result = run_backtest(bars, scores, tuned, strategy=strategy)
        row = result.report.as_row()
        row["threshold"] = threshold
        rows.append(row)
    return pd.DataFrame(rows).set_index("threshold")


def breakeven_threshold(
    bars: pd.DataFrame,
    scores: pd.Series,
    config: BacktestConfig,
    low: float = 0.0,
    high: float = 1.0,
    steps: int = 12,
) -> float | None:
    """Smallest entry threshold with a positive net expectancy, if any."""
    best: float | None = None
    for threshold in np.linspace(low, high, steps):
        tuned = BacktestConfig(
            horizon=config.horizon,
            entry_threshold=float(threshold),
            notional=config.notional,
            lot_size=config.lot_size,
            cost_model=config.cost_model,
        )
        result = run_backtest(bars, scores, tuned)
        if result.report.expectancy_bps > 0:
            best = float(threshold)
            break
    return best


def session_exit_time(stamp) -> bool:
    """Whether a timestamp is at or past the final minute of the session."""
    moment = stamp.astimezone(IST) if hasattr(stamp, "astimezone") else stamp
    return moment.hour == 15 and moment.minute >= 29
