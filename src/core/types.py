"""Core domain types shared across the platform.

These are the nouns every plugin agrees on. A source produces ``Tick``s, an
aggregator turns them into bars, a strategy emits a ``Signal``, a forecaster a
``Prediction`` and a ``ForecastCandle``, and a renderer draws a
``MarketSnapshot``. Because the vocabulary lives here rather than in any plugin,
plugins can be added, replaced, or removed without a single change to the types
they exchange.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

import pandas as pd


class Direction(StrEnum):
    """Predicted or observed market direction."""

    UP = "UP"
    DOWN = "DOWN"
    FLAT = "FLAT"

    @property
    def sign(self) -> int:
        return {Direction.UP: 1, Direction.DOWN: -1, Direction.FLAT: 0}[self]

    @classmethod
    def from_value(cls, value: float, deadband: float = 0.0) -> Direction:
        if value > deadband:
            return cls.UP
        if value < -deadband:
            return cls.DOWN
        return cls.FLAT


@dataclass(slots=True)
class Tick:
    """A single market data update."""

    ts: datetime
    ltp: float
    ltq: int = 0
    close_prev: float = 0.0
    bid_p: float = 0.0
    bid_q: int = 0
    ask_p: float = 0.0
    ask_q: int = 0
    volume_traded: int = 0
    avg_traded_price: float = 0.0
    open_interest: float = 0.0
    total_buy_qty: float = 0.0
    total_sell_qty: float = 0.0
    instrument_key: str = ""
    greeks: dict = field(default_factory=dict)

    @property
    def mid(self) -> float:
        if self.bid_p and self.ask_p:
            return (self.bid_p + self.ask_p) / 2.0
        return self.ltp

    @property
    def spread(self) -> float:
        if self.bid_p and self.ask_p:
            return self.ask_p - self.bid_p
        return 0.0

    @property
    def depth_imbalance(self) -> float:
        """Order book imbalance in [-1, 1]; positive means bid-heavy."""
        total = self.bid_q + self.ask_q
        if total <= 0:
            return 0.0
        return (self.bid_q - self.ask_q) / total

    @property
    def trade_imbalance(self) -> float:
        """Aggregate traded-value imbalance in [-1, 1]; positive means buy-heavy."""
        total = self.total_buy_qty + self.total_sell_qty
        if total <= 0:
            return 0.0
        return (self.total_buy_qty - self.total_sell_qty) / total


@dataclass(slots=True)
class Signal:
    """A directional opinion from a single strategy at a point in time."""

    ts: datetime
    strategy: str
    direction: Direction
    strength: float
    reason: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def score(self) -> float:
        """Signed conviction in [-1, 1]."""
        return self.direction.sign * float(self.strength)


@dataclass(slots=True)
class Prediction:
    """Model output for one forecast horizon."""

    ts: datetime
    horizon_min: int
    p_up: float
    direction: Direction
    confidence: float
    model: str = "ensemble"
    expected_move_bps: float = 0.0
    hurdle_bps: float = 0.0
    contributions: dict = field(default_factory=dict)

    @property
    def edge(self) -> float:
        """Distance from an uninformative 50/50 forecast."""
        return abs(self.p_up - 0.5) * 2.0

    @property
    def clears_hurdle(self) -> bool:
        """Whether the forecast move is large enough to pay for the round trip.

        This is the question that decides whether a scalping signal is worth
        acting on at all. A directional forecast that is right but forecasts a
        move smaller than transaction costs loses money with every correct call,
        so the horizon is only tradeable when the expected move beats the cost.
        """
        return abs(self.expected_move_bps) > self.hurdle_bps

    @property
    def edge_after_cost_bps(self) -> float:
        """Expected move net of the round-trip hurdle; negative means untradeable."""
        return abs(self.expected_move_bps) - self.hurdle_bps

    @property
    def member_disagreement(self) -> float:
        """Dispersion of model probabilities in [0, 0.5]."""
        import math

        values = [
            float(value)
            for value in self.contributions.values()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        ]
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5

    @property
    def member_agreement(self) -> float:
        """A UI-friendly 0..1 agreement score; one means members coincide."""
        return max(0.0, 1.0 - min(self.member_disagreement * 2.0, 1.0))


@dataclass(slots=True)
class ForecastCandle:
    """A projected OHLC bar sitting after the forming one.

    Not a prediction of a specific price path — an estimate of where the next
    few bars are likely to travel, given current conviction and volatility. It is
    rebuilt continuously from the live price, so a projected candle moves as the
    market moves underneath it rather than being fixed at the moment it was made.

    ``horizon`` counts complete bars after the current/forming bar. Timestamps
    are bar OPEN times; a target is scored after that entire bar closes.
    """

    ts: datetime
    horizon: int
    open: float
    high: float
    low: float
    close: float
    conviction: float = 0.0
    confidence: float = 0.0
    expected_move_bps: float = 0.0
    bar_minutes: int = 1

    @property
    def direction(self) -> Direction:
        return Direction.from_value(self.close - self.open)

    @property
    def is_up(self) -> bool:
        return self.close >= self.open

    @property
    def change_bps(self) -> float:
        if not self.open:
            return 0.0
        return (self.close / self.open - 1.0) * 10_000

    def as_row(self) -> dict:
        """The bar as a row of the OHLCV schema, for charting."""
        return {
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": 0.0,
            "oi": 0.0,
        }


@dataclass(slots=True)
class LegVerdict:
    """What one leg of the board is worth doing, and the arithmetic behind it.

    Every number here is derived from the same three questions the rest of the
    platform asks: what does the move have to be, what is it projected to be, and
    what does the position pay while waiting. A verdict without its arithmetic
    would be an opinion; with it, a reader can disagree with the inputs.
    """

    label: str
    action: str
    required_move_bps: float
    projected_move_bps: float
    theta_cost_bps: float = 0.0
    cost_bps: float = 0.0
    iv: float = 0.0
    realised_vol: float = 0.0
    reason: str = ""

    @property
    def edge_bps(self) -> float:
        return abs(self.projected_move_bps) - self.required_move_bps

    @property
    def is_trade(self) -> bool:
        return self.action != "FLAT"

    def as_row(self) -> dict:
        return {
            "label": self.label,
            "action": self.action,
            "required_bps": round(self.required_move_bps, 1),
            "projected_bps": round(self.projected_move_bps, 1),
            "edge_bps": round(self.edge_bps, 1),
            "theta_bps": round(self.theta_cost_bps, 1),
            "iv": round(self.iv, 4),
            "reason": self.reason,
        }


@dataclass(slots=True)
class LegSnapshot:
    """One instrument on the board: the index, a call, or a put."""

    label: str
    kind: str
    snapshot: MarketSnapshot
    strike: float = 0.0
    greeks: dict = field(default_factory=dict)
    verdict: LegVerdict | None = None

    @property
    def is_option(self) -> bool:
        return self.kind in {"CE", "PE"}


@dataclass(slots=True)
class BoardSnapshot:
    """Everything the renderer needs for a call/index/put frame."""

    ts: datetime
    symbol: str
    spot: float
    legs: list[LegSnapshot] = field(default_factory=list)
    chain: dict = field(default_factory=dict)
    headline: str = ""
    note: str = ""

    def leg(self, label: str) -> LegSnapshot | None:
        for item in self.legs:
            if item.label == label:
                return item
        return None

    @property
    def spot_leg(self) -> LegSnapshot | None:
        return self.leg("INDEX")

    def option_legs(self) -> list[LegSnapshot]:
        return [item for item in self.legs if item.is_option]

    def tradeable(self) -> list[LegVerdict]:
        return [item.verdict for item in self.legs if item.verdict and item.verdict.is_trade]


@dataclass
class MarketSnapshot:
    """Everything the UI needs to render one frame."""

    ts: datetime
    symbol: str
    last_price: float
    prev_close: float
    candles: pd.DataFrame
    signals: list[Signal] = field(default_factory=list)
    predictions: list[Prediction] = field(default_factory=list)
    regime: str = "unknown"
    indicators: dict = field(default_factory=dict)
    projections: list[ForecastCandle] = field(default_factory=list)
    conviction: float = 0.0
    projection_ts: datetime | None = None
    projection_error_bps: float = 0.0
    research: dict = field(default_factory=dict)
    source: str = "simulation"
    instrument_key: str = ""
    last_tick_ts: datetime | None = None

    @property
    def change(self) -> float:
        return self.last_price - self.prev_close

    @property
    def change_pct(self) -> float:
        if not self.prev_close:
            return 0.0
        return (self.last_price / self.prev_close - 1.0) * 100.0


@dataclass
class Trade:
    """A single round-trip trade produced by the backtester."""

    entry_ts: datetime
    exit_ts: datetime
    direction: Direction
    entry_price: float
    exit_price: float
    quantity: int = 1
    costs: float = 0.0
    strategy: str = ""

    @property
    def gross_pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.direction.sign * self.quantity

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.costs

    @property
    def return_bps(self) -> float:
        if not self.entry_price:
            return 0.0
        return (self.exit_price / self.entry_price - 1.0) * self.direction.sign * 10_000
