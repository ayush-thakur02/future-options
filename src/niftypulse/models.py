"""Core domain types shared across the platform."""

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
