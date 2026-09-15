"""One open lot, and the rules that close it.

A position is always exactly one lot of the leg's contract — 65 units by default.
That is not a sizing decision the simulator makes; it is the smallest thing that
can be traded and, at these balances, the largest thing too.

Marking to market is the leg's **own traded price**: the index level for the
index, the live premium for an option. Nothing here reprices an option from a
model, so the P&L a reader sees is the P&L the tape printed rather than the
output of the same pricing function that drew the chart.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from core.types import Direction

LONG = "LONG"
SHORT = "SHORT"


@dataclass(slots=True)
class Position:
    """A single open lot on one leg."""

    leg: str
    kind: str
    instrument: str
    direction: Direction
    units: int
    entry_price: float
    entry_ts: datetime
    entry_cost: float
    view: float = 0.0
    edge_bps: float = 0.0
    reason: str = ""
    stop_price: float = 0.0
    target_price: float = 0.0
    expires_at: datetime | None = None

    @property
    def side(self) -> int:
        return self.direction.sign

    @property
    def is_long(self) -> bool:
        return self.direction is Direction.UP

    @property
    def notional(self) -> float:
        """Rupees of exposure the lot carries, which is what costs are quoted on."""
        return max(self.entry_price * self.units, 0.0)

    def unrealized(self, price: float) -> float:
        """Gross mark-to-market, before the cost of getting out."""
        if price <= 0 or self.entry_price <= 0:
            return 0.0
        return (float(price) - self.entry_price) * self.side * self.units

    def return_bps(self, price: float) -> float:
        if self.entry_price <= 0 or price <= 0:
            return 0.0
        return (float(price) / self.entry_price - 1.0) * self.side * 10_000.0

    def age_seconds(self, now: datetime) -> float:
        return max((now - self.entry_ts).total_seconds(), 0.0)

    def hold_minutes(self, now: datetime) -> float:
        return self.age_seconds(now) / 60.0

    # ------------------------------------------------------------------ exits

    def breached(self, price: float) -> str:
        """Whether the stop or the target has been touched. Empty string if not.

        Both levels are checked against the same print, so a bar that gaps through
        the stop cannot also be read as having reached the target first.
        """
        if price <= 0:
            return ""
        if self.stop_price > 0:
            if (self.is_long and price <= self.stop_price) or (not self.is_long and price >= self.stop_price):
                return "stop"
        if self.target_price > 0:
            if (self.is_long and price >= self.target_price) or (not self.is_long and price <= self.target_price):
                return "target"
        return ""

    def expired(self, now: datetime) -> bool:
        return self.expires_at is not None and now >= self.expires_at

    def exit_reason(self, price: float, view: float, now: datetime, flip: float = -0.2) -> str:
        """Why to close, in priority order. Empty string means keep holding."""
        touched = self.breached(price)
        if touched:
            return touched
        if self.expired(now):
            return "time"
        # The signal that opened the position has to keep pointing the same way.
        # A flip is judged on the blend the entry was taken on, not on price
        # alone, so a position is not closed by its own adverse excursion before
        # the stop has had a chance to be the thing that closes it.
        if view * self.side <= flip:
            return "reversal"
        return ""

    def as_row(self, price: float, now: datetime) -> dict:
        return {
            "leg": self.leg,
            "side": LONG if self.is_long else SHORT,
            "units": self.units,
            "entry_price": round(self.entry_price, 2),
            "price": round(float(price), 2),
            "entry_ts": self.entry_ts.isoformat(),
            "entry_cost": round(self.entry_cost, 2),
            "unrealized": round(self.unrealized(price), 2),
            "return_bps": round(self.return_bps(price), 2),
            "stop_price": round(self.stop_price, 2),
            "target_price": round(self.target_price, 2),
            "hold_minutes": round(self.hold_minutes(now), 2),
            "view": round(self.view, 4),
            "edge_bps": round(self.edge_bps, 2),
            "reason": self.reason,
        }


def levels_for(
    direction: Direction,
    price: float,
    atr: float,
    stop_atr: float,
    target_atr: float,
    fallback_bps: float = 10.0,
) -> tuple[float, float]:
    """Stop and target prices for a new lot.

    Volatility-scaled when an ATR is available, and a fixed basis-point distance
    when it is not. The fallback matters more than it looks: a leg whose history
    has not warmed up yet reports an ATR of zero, and a stop at the entry price
    would close every position on the next print.
    """
    base = float(price)
    if base <= 0:
        return 0.0, 0.0
    span = float(atr) if atr and atr > 0 else base * fallback_bps / 10_000.0
    risk = span * max(float(stop_atr), 0.0)
    reward = span * max(float(target_atr), 0.0)
    if direction is Direction.UP:
        return base - risk, base + reward
    return base + risk, base - reward


def expiry_of(entry: datetime, max_hold_minutes: float) -> datetime:
    return entry + timedelta(minutes=max(float(max_hold_minutes), 0.0))


__all__ = ["LONG", "SHORT", "Position", "expiry_of", "levels_for"]
