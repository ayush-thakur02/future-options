"""Wallets, and the reserve that keeps them trading.

Two pieces of arithmetic live here, and they are the whole reason the simulator
can be read as a risk statement rather than as a score.

A :class:`Wallet` holds **risk capital**, not notional. It starts at its opening
balance and moves only when a position closes, by the rupees that position made
or lost. A 65-unit index leg is roughly Rs 1.5 million of exposure and never
appears in the wallet, because the wallet is not funding it — it is absorbing the
Rs 65 per point that the position is worth.

A :class:`Reserve` is a shared pool drawn on when a wallet has lost enough that
it can no longer fund a lot. Draws are repaid from profits before those profits
count as the leg's own, which makes the reserve temporary in fact and not only
in intention: a leg that keeps losing eventually exhausts it, and a leg that
recovers gives it back.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Wallet:
    """One leg's risk capital, its history, and what it has borrowed."""

    label: str
    opening: float
    cash: float
    reserve_drawn: float = 0.0
    realized: float = 0.0
    costs_paid: float = 0.0
    trades: int = 0
    wins: int = 0
    losses: int = 0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    peak_equity: float = 0.0
    top_ups: int = 0

    def __post_init__(self) -> None:
        self.opening = float(self.opening)
        self.cash = float(self.cash)
        self.peak_equity = max(float(self.peak_equity), self.cash)

    # ------------------------------------------------------------------ money

    def charge(self, amount: float) -> None:
        """Pay a cost now, before any result is known. The entry leg of a trade."""
        value = max(float(amount), 0.0)
        self.cash -= value
        self.costs_paid += value

    def settle(self, cash_delta: float, net_pnl: float, costs: float) -> None:
        """Apply a closed trade: the rupees that arrived, and the record of them.

        ``cash_delta`` and ``net_pnl`` are passed separately because they differ
        by whatever was already charged at entry. The wallet moved by the exit
        leg only, while what the trade made is measured against both legs — and
        conflating the two would either double-count the entry cost or hide it.
        """
        net = float(net_pnl)
        self.cash += float(cash_delta)
        self.realized += net
        self.costs_paid += max(float(costs), 0.0)
        self.trades += 1
        self.best_trade = max(self.best_trade, net)
        self.worst_trade = min(self.worst_trade, net)
        if net > 0:
            self.wins += 1
        else:
            self.losses += 1

    def equity(self, unrealized: float = 0.0) -> float:
        return self.cash + float(unrealized)

    def observe(self, unrealized: float = 0.0) -> float:
        """Mark the wallet and keep the running peak. Returns current equity."""
        current = self.equity(unrealized)
        self.peak_equity = max(self.peak_equity, current)
        return current

    # ------------------------------------------------------------ diagnostics

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades else 0.0

    @property
    def total_pnl(self) -> float:
        return self.cash - self.opening

    @property
    def return_pct(self) -> float:
        if not self.opening:
            return 0.0
        return (self.cash / self.opening - 1.0) * 100.0

    def drawdown(self, unrealized: float = 0.0) -> float:
        return max(self.peak_equity - self.equity(unrealized), 0.0)

    def as_row(self, unrealized: float = 0.0) -> dict:
        return {
            "label": self.label,
            "opening": round(self.opening, 2),
            "cash": round(self.cash, 2),
            "equity": round(self.equity(unrealized), 2),
            "unrealized": round(float(unrealized), 2),
            "realized": round(self.realized, 2),
            "total_pnl": round(self.total_pnl, 2),
            "return_pct": round(self.return_pct, 3),
            "costs_paid": round(self.costs_paid, 2),
            "reserve_drawn": round(self.reserve_drawn, 2),
            "top_ups": self.top_ups,
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 4),
            "best_trade": round(self.best_trade, 2),
            "worst_trade": round(self.worst_trade, 2),
            "drawdown": round(self.drawdown(unrealized), 2),
        }


@dataclass
class Reserve:
    """The shared pool a drawn-down wallet borrows from.

    Shared rather than split three ways on purpose: a leg that is working should
    be able to keep going when its neighbour has stopped, and a fixed per-leg
    slice would strand capital in the leg that is losing.
    """

    opening: float
    remaining: float = 0.0
    lent: dict[str, float] = field(default_factory=dict)
    calls: int = 0

    def __post_init__(self) -> None:
        self.opening = float(self.opening)
        self.remaining = float(self.remaining or self.opening)

    @property
    def deployed(self) -> float:
        return self.opening - self.remaining

    @property
    def deployed_pct(self) -> float:
        return (self.deployed / self.opening * 100.0) if self.opening else 0.0

    def can_lend(self, amount: float = 1.0) -> bool:
        return self.remaining >= min(float(amount), 1e-9) and self.remaining > 0.0

    def expose(self, leg: str) -> float:
        return float(self.lent.get(leg, 0.0))

    def lend(self, leg: str, amount: float) -> float:
        """Move rupees from the pool to a wallet. Returns what was actually lent."""
        want = min(max(float(amount), 0.0), self.remaining)
        if want <= 0.0:
            return 0.0
        self.remaining -= want
        self.lent[leg] = self.lent.get(leg, 0.0) + want
        self.calls += 1
        return want

    def repay(self, leg: str, amount: float) -> float:
        """Take rupees back from a wallet. Returns what was actually repaid."""
        owed = float(self.lent.get(leg, 0.0))
        got = min(max(float(amount), 0.0), owed)
        if got <= 0.0:
            return 0.0
        self.remaining += got
        self.lent[leg] = owed - got
        return got

    def as_row(self) -> dict:
        return {
            "opening": round(self.opening, 2),
            "remaining": round(self.remaining, 2),
            "deployed": round(self.deployed, 2),
            "deployed_pct": round(self.deployed_pct, 3),
            "calls": self.calls,
            "by_leg": {leg: round(value, 2) for leg, value in sorted(self.lent.items())},
        }


def round_to_rupee(value: float) -> float:
    """Simulated money is quoted in whole rupees; sub-paisa precision is noise."""
    if not math.isfinite(float(value)):
        return 0.0
    return float(round(float(value), 2))


__all__ = ["Reserve", "Wallet", "round_to_rupee"]
