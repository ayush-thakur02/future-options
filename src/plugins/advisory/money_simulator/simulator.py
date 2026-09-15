"""The money simulator: three wallets, real prices, rupees.

One decision loop, run against the tape that is actually arriving. Each leg holds
at most one lot; the loop marks it, closes it when a rule says to, and otherwise
asks the policy whether the balance of everything the platform knows has moved far
enough to justify a new one.

Two cadences, and the split matters. **Risk is checked on every refresh** — a stop
is a price level, and a stop that is only tested once a minute is not a stop, it
is a hope. **Entries are checked on the decision clock**, thirty seconds by
default, because an entry is a judgement about the next few minutes and re-taking
the same judgement every second buys nothing but noise.

The accounting is a cash account over risk capital. A lot's rupees of exposure
never enter the wallet; what enters is the rupees-per-point it is worth. A long
index leg at 24,000 in a 25,000 wallet is not a 1.5-million-rupee position the
wallet cannot afford — it is a claim on 65 rupees a point, and the wallet is what
absorbs being wrong about it.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime

from core.types import Direction

from .ledger import ClosedTrade, SimulationLedger
from .policy import (
    BUY,
    EXIT,
    HOLD,
    Decision,
    DecisionPolicy,
    LegState,
    LegView,
    PolicyWeights,
    as_leg_state,
)
from .position import LONG, SHORT, Position, expiry_of, levels_for
from .wallet import Reserve, Wallet

# Below this the wallet cannot fund another lot and the reserve is asked for one.
# Half the opening balance rather than a fixed rupee figure, so the level means
# the same thing whatever the account is funded with.
DEFAULT_MIN_EQUITY_FRACTION = 0.5


@dataclass
class SimulatorConfig:
    """Everything about the simulator that a reader might want to disagree with."""

    opening_balance: float = 25_000.0
    reserve: float = 200_000.0
    lot_size: int = 65
    # How often entries are considered. Risk runs on every tick, not on this.
    decision_seconds: float = 10.0
    # An approved entry waits for a price to print against. If none does within
    # this long the view that justified it has gone stale and the order is pulled.
    entry_timeout_seconds: float = 30.0
    horizon_bars: int = 3
    entry_view: float = 0.15
    min_edge_bps: float = 0.0
    min_confirmations: int = 2
    require_projection: bool = True
    min_equity_fraction: float = DEFAULT_MIN_EQUITY_FRACTION
    stop_atr: float = 1.5
    target_atr: float = 2.5
    fallback_stop_bps: float = 10.0
    max_hold_minutes: float = 5.0
    min_hold_seconds: float = 45.0
    cooldown_seconds: float = 45.0
    reversal_view: float = -0.15
    option_cost_rate: float = 0.012
    projection_scale_bps: float = 4.0
    weights: PolicyWeights = field(default_factory=PolicyWeights)

    def as_row(self) -> dict:
        return {
            "opening_balance": self.opening_balance,
            "reserve": self.reserve,
            "lot_size": self.lot_size,
            "decision_seconds": self.decision_seconds,
            "entry_timeout_seconds": self.entry_timeout_seconds,
            "horizon_bars": self.horizon_bars,
            "entry_view": self.entry_view,
            "min_edge_bps": self.min_edge_bps,
            "min_confirmations": self.min_confirmations,
            "require_projection": self.require_projection,
            "min_equity_fraction": self.min_equity_fraction,
            "stop_atr": self.stop_atr,
            "target_atr": self.target_atr,
            "max_hold_minutes": self.max_hold_minutes,
            "min_hold_seconds": self.min_hold_seconds,
            "cooldown_seconds": self.cooldown_seconds,
            "option_cost_rate": self.option_cost_rate,
            "weights": self.weights.as_row(),
        }


@dataclass(slots=True)
class _Intent:
    """An entry the policy has approved and the tape has not yet priced.

    Held rather than filled on the spot, because the price a decision was made on
    is not a price anything can be executed at — the platform's own backtester
    enters on the next bar's open for the same reason. It also gives the order a
    moment in which the tape can invalidate it.
    """

    leg: str
    kind: str
    instrument: str
    action: str
    view: float
    edge_bps: float
    reason: str
    atr: float
    created_at: datetime


class MoneySimulator:
    """Runs a funded paper book across the index, the call and the put."""

    def __init__(
        self,
        config: SimulatorConfig | None = None,
        *,
        ledger: SimulationLedger | None = None,
        cost_model=None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config or SimulatorConfig()
        self.ledger = ledger
        self.cost_model = cost_model
        self.log = logger or logging.getLogger("niftypulse.money")
        self._lock = threading.RLock()

        self.wallets: dict[str, Wallet] = {}
        self.reserve = Reserve(opening=self.config.reserve)
        self.positions: dict[str, Position] = {}
        self._intents: dict[str, _Intent] = {}
        self.views: dict[str, LegView] = {}
        self.decisions: deque[Decision] = deque(maxlen=90)
        self.closed: deque[ClosedTrade] = deque(maxlen=120)
        self.started_at: datetime | None = None
        self.updated_at: datetime | None = None
        self.steps = 0
        self.last_decision_at: datetime | None = None
        self._cooldown_until: dict[str, datetime] = {}
        self.policy = DecisionPolicy(
            weights=self.config.weights,
            horizon_bars=self.config.horizon_bars,
            entry_view=self.config.entry_view,
            min_edge_bps=self.config.min_edge_bps,
            min_confirmations=self.config.min_confirmations,
            option_cost_rate=self.config.option_cost_rate,
            projection_scale_bps=self.config.projection_scale_bps,
            units=self.config.lot_size,
            require_projection=self.config.require_projection,
        )
        self._restore()

    # ------------------------------------------------------------- lifecycle

    def _restore(self) -> None:
        """Pick up where the last run stopped, if the ledger has anything to say."""
        if self.ledger is None:
            return
        state = self.ledger.restore(self.config.opening_balance)
        if state.reserve is not None:
            self.reserve = state.reserve
        self.wallets = state.wallets

    def reset(self) -> None:
        """Hand every leg its opening balance back and empty the reserve."""
        with self._lock:
            self.wallets = {}
            self.reserve = Reserve(opening=self.config.reserve)
            self.positions.clear()
            self._intents.clear()
            self.views.clear()
            self.decisions.clear()
            self.closed.clear()
            self._cooldown_until.clear()
            self.last_decision_at = None
            if self.ledger is not None:
                self.ledger.reset()

    def _wallet(self, leg: str) -> Wallet:
        wallet = self.wallets.get(leg)
        if wallet is None:
            wallet = Wallet(label=leg, opening=self.config.opening_balance, cash=self.config.opening_balance)
            self.wallets[leg] = wallet
        return wallet

    # ------------------------------------------------------------------ clock

    def step(self, legs, now: datetime, *, session_open: bool = True) -> list[Decision]:
        """Fill what is waiting, close what must close, and judge what may enter.

        Called on the refresh clock. Exits run every call; entries are *decided*
        on the decision clock and *filled* on the next price that prints, which is
        what :meth:`mark` is for.

        ``legs`` accepts either :class:`~.policy.LegState` records or the plain
        mappings the board produces, so the runtime never has to import this
        pack's types to drive it.
        """
        with self._lock:
            states = [as_leg_state(item) for item in legs]
            prices = {state.label: float(state.price or 0.0) for state in states}
            if self.started_at is None:
                self.started_at = now
            self.steps += 1
            self.updated_at = now

            if not session_open:
                self._intents.clear()
                produced = self._flatten(prices, now, "session closed")
                self._mark(prices)
                return produced

            produced = self._fill_intents(prices, now)
            produced.extend(self._manage(prices, now))
            due = (
                self.last_decision_at is None
                or (now - self.last_decision_at).total_seconds() >= self.config.decision_seconds
            )
            if due:
                self.last_decision_at = now
                produced.extend(self._decide(states, now))
            self._mark(prices)
            return produced

    def mark(self, prices: Mapping[str, float], now: datetime) -> list[Decision]:
        """React to one print, without re-judging anything.

        This is the tick path. A stop is a price level, and a level tested once a
        second is a level the tape gets to cross and come back from in between —
        so risk is checked on every print the feed delivers, while the decision to
        open a position stays on its own slower clock.
        """
        with self._lock:
            if not prices:
                return []
            self.updated_at = now
            produced = self._fill_intents(prices, now)
            produced.extend(self._manage(prices, now))
            self._mark(prices)
            return produced

    # ---------------------------------------------------------------- entries

    def _fill_intents(self, prices: Mapping[str, float], now: datetime) -> list[Decision]:
        """Execute approved entries at a price that has actually printed."""
        produced: list[Decision] = []
        for leg, intent in list(self._intents.items()):
            if now <= intent.created_at:
                continue
            if leg in self.positions:
                del self._intents[leg]
                continue
            price = float(prices.get(leg, 0.0) or 0.0)
            expired = (now - intent.created_at).total_seconds() > self.config.entry_timeout_seconds
            if expired:
                # Checked before the price, so an order is pulled on its own clock
                # rather than waiting for a tape that may have gone dark.
                del self._intents[leg]
                produced.append(
                    self._record_intent(
                        intent,
                        price,
                        now,
                        f"order pulled — nothing filled it in "
                        f"{self.config.entry_timeout_seconds:.0f}s",
                    )
                )
                continue
            if price <= 0:
                continue
            del self._intents[leg]
            produced.append(self._open(intent, price, now))
        return produced

    def _open(self, intent: _Intent, price: float, now: datetime) -> Decision:
        """Fund one lot and put it on, at the price the tape gave."""
        self._ensure_capital(intent.leg)
        wallet = self._wallet(intent.leg)
        units = self.config.lot_size
        direction = Direction.UP if intent.action == BUY else Direction.DOWN

        entry_cost = self._cost(
            intent.kind, price, units, "buy" if direction is Direction.UP else "sell"
        )
        if wallet.cash - entry_cost <= 0:
            return self._record_intent(
                intent, price, now, "wallet exhausted and the reserve is spent"
            )
        wallet.charge(entry_cost)

        stop, target = levels_for(
            direction,
            price,
            intent.atr,
            self.config.stop_atr,
            self.config.target_atr,
            fallback_bps=self.config.fallback_stop_bps,
        )
        position = Position(
            leg=intent.leg,
            kind=intent.kind,
            instrument=intent.instrument,
            direction=direction,
            units=units,
            entry_price=price,
            entry_ts=now,
            entry_cost=entry_cost,
            view=intent.view,
            edge_bps=intent.edge_bps,
            reason=intent.reason,
            stop_price=stop,
            target_price=target,
            expires_at=expiry_of(now, self.config.max_hold_minutes),
        )
        self.positions[intent.leg] = position
        self._persist()
        decision = Decision(
            leg=intent.leg,
            action=intent.action,
            reason=intent.reason,
            view=intent.view,
            price=price,
            at=now,
            edge_bps=intent.edge_bps,
            units=units,
        )
        self.decisions.append(decision)
        return decision

    # ----------------------------------------------------------------- exits

    def _manage(self, prices: Mapping[str, float], now: datetime) -> list[Decision]:
        """Close anything the tape has resolved. Runs on every print."""
        produced: list[Decision] = []
        for label, position in list(self.positions.items()):
            price = float(prices.get(label, 0.0) or 0.0)
            if price <= 0:
                continue
            view = self.views.get(label)
            blended = view.view if view is not None else 0.0
            reason = position.exit_reason(price, blended, now, flip=self.config.reversal_view)
            # A reversal is the one exit that is a judgement rather than a price
            # level, so it waits out the minimum hold. A stop does not: the whole
            # point of a stop is that it acts the moment it is touched.
            if reason == "reversal" and not self._held_long_enough(position, now):
                reason = ""
            if reason:
                produced.append(self._close(position, price, now, reason))
                continue
            wallet = self._wallet(label)
            if wallet.equity(position.unrealized(price)) <= 0.0:
                produced.append(self._close(position, price, now, "ruin"))
        return produced

    def _held_long_enough(self, position: Position, now: datetime) -> bool:
        return position.age_seconds(now) >= self.config.min_hold_seconds

    def _close(
        self,
        position: Position,
        price: float,
        now: datetime,
        reason: str,
    ) -> Decision:
        """Settle one lot at the tape, charge the exit, and book the result.

        A stop and a target are resting orders, so they fill **at their level** —
        filling at whatever the next print happened to be would let a two-second
        gap between refreshes turn a 20-point stop into a 300-point loss and make
        every stop look like a catastrophe. Time, reversal and session exits are
        market orders and fill at the tape.
        """
        fill = self._fill_price(position, price, reason)
        gross = position.unrealized(fill)
        exit_cost = self._cost(
            position.kind,
            fill,
            position.units,
            "sell" if position.is_long else "buy",
        )
        total_costs = position.entry_cost + exit_cost
        # The entry cost already left the wallet when the lot was opened, so the
        # cash that arrives at the exit is the gross less the exit leg only. What
        # the trade *made* is measured against both legs, and the two are passed
        # to the wallet separately rather than conflated.
        net = gross - total_costs
        wallet = self._wallet(position.leg)
        wallet.settle(cash_delta=gross - exit_cost, net_pnl=net, costs=exit_cost)
        trade = ClosedTrade(
            leg=position.leg,
            kind=position.kind,
            instrument=position.instrument,
            side=LONG if position.is_long else SHORT,
            units=position.units,
            entry_ts=position.entry_ts,
            entry_price=position.entry_price,
            exit_ts=now,
            exit_price=float(fill),
            gross=gross,
            costs=total_costs,
            net=net,
            return_bps=position.return_bps(fill),
            exit_reason=reason,
            reason=position.reason,
            view=position.view,
            edge_bps=position.edge_bps,
        )
        self.positions.pop(position.leg, None)
        self.closed.append(trade)
        if self.ledger is not None:
            self.ledger.record(trade)
        self._cooldown_until[position.leg] = now
        self._sweep(position.leg)
        self._persist()
        decision = Decision(
            leg=position.leg,
            action=EXIT,
            reason=f"{reason} · {trade.net:+,.0f}",
            view=position.view,
            price=float(fill),
            at=now,
            edge_bps=position.edge_bps,
            units=position.units,
            pnl=trade.net,
        )
        self.decisions.append(decision)
        return decision

    def _fill_price(self, position: Position, price: float, reason: str) -> float:
        """Where the order actually fills: the level if it rests, the tape if not."""
        if reason == "stop" and position.stop_price > 0:
            return float(position.stop_price)
        if reason == "target" and position.target_price > 0:
            return float(position.target_price)
        return float(price)

    def _flatten(self, prices: Mapping[str, float], now: datetime, reason: str) -> list[Decision]:
        """Close everything — the session is over, and nothing is carried overnight."""
        produced: list[Decision] = []
        for label in list(self.positions):
            price = float(prices.get(label, 0.0) or 0.0)
            if price <= 0:
                continue
            produced.append(self._close(self.positions[label], price, now, reason))
        return produced

    # --------------------------------------------------------------- entries

    def _decide(self, legs: list[LegState], now: datetime) -> list[Decision]:
        produced: list[Decision] = []
        for state in legs:
            view = self.policy.view_for(state, self.cost_model)
            self.views[state.label] = view

            if state.label in self.positions or state.label in self._intents:
                continue

            action, reason = self.policy.action_for(view, has_position=False)
            if action is HOLD:
                produced.append(self._record(state, HOLD, reason, view, now))
                continue

            blocked = self._blocked(state, now)
            if blocked:
                produced.append(self._record(state, HOLD, blocked, view, now))
                continue

            if float(state.price or 0.0) <= 0:
                continue

            # Approved, not filled. The order waits for a price the tape actually
            # prints — the one this judgement was made on is gone by now.
            self._intents[state.label] = _Intent(
                leg=state.label,
                kind=state.kind,
                instrument=state.instrument,
                action=action,
                view=view.view,
                edge_bps=view.edge_bps,
                reason=reason,
                atr=self._atr_of(state),
                created_at=now,
            )
            produced.append(
                self._record(state, action, f"{reason} · order working", view, now)
            )
        return produced

    def _record_intent(
        self, intent: _Intent, price: float, now: datetime, reason: str
    ) -> Decision:
        decision = Decision(
            leg=intent.leg,
            action=HOLD,
            reason=reason,
            view=intent.view,
            price=price,
            at=now,
            edge_bps=intent.edge_bps,
        )
        self.decisions.append(decision)
        return decision

    def _blocked(self, state: LegState, now: datetime) -> str:
        """Entry refusals that have nothing to do with the view."""
        since = self._cooldown_until.get(state.label)
        if since is not None and (now - since).total_seconds() < self.config.cooldown_seconds:
            remaining = self.config.cooldown_seconds - (now - since).total_seconds()
            return f"cooling down {remaining:.0f}s after the last close"
        return ""

    def _record(
        self, state: LegState, action: str, reason: str, view: LegView, now: datetime
    ) -> Decision:
        decision = Decision(
            leg=state.label,
            action=action,
            reason=reason,
            view=view.view,
            price=float(state.price or 0.0),
            at=now,
            edge_bps=view.edge_bps,
        )
        self.decisions.append(decision)
        return decision

    # ----------------------------------------------------------- the reserve

    def _ensure_capital(self, leg: str) -> None:
        """Top a drawn-down wallet back up, so the leg can keep investing."""
        wallet = self._wallet(leg)
        floor = self.config.opening_balance * self.config.min_equity_fraction
        if wallet.cash >= floor:
            return
        want = self.config.opening_balance - wallet.cash
        got = self.reserve.lend(leg, want)
        if got <= 0:
            return
        wallet.cash += got
        wallet.reserve_drawn += got
        wallet.top_ups += 1
        self.log.info(
            "%s topped up by %.0f from the reserve (%.0f left)",
            leg,
            got,
            self.reserve.remaining,
        )

    def _sweep(self, leg: str) -> None:
        """Give the reserve its money back before profits count as the leg's own."""
        wallet = self._wallet(leg)
        if wallet.reserve_drawn <= 0:
            return
        excess = wallet.cash - self.config.opening_balance
        if excess <= 0:
            return
        repaid = self.reserve.repay(leg, min(excess, wallet.reserve_drawn))
        if repaid <= 0:
            return
        wallet.cash -= repaid
        wallet.reserve_drawn -= repaid

    def force_top_up(self, leg: str) -> float:
        """Top one leg up now, whatever its balance. Used by tests and tooling."""
        with self._lock:
            self._ensure_capital(leg)
            return float(self._wallet(leg).cash)

    # --------------------------------------------------------------- costing

    def _cost(self, kind: str, price: float, units: int, side: str) -> float:
        """One side of a round trip, in rupees.

        Options are charged as a fraction of the premium, which is where an
        option's cost actually lives — spread and STT on a ten-thousand-rupee
        premium are nothing like the basis points the same trade would pay as
        futures. The index is charged through the platform's own cost model, so
        the simulator pays exactly what the rest of the platform says a scalp
        costs.
        """
        notional = max(float(price) * int(units), 0.0)
        if notional <= 0:
            return 0.0
        if kind in {"CE", "PE"}:
            return notional * self.config.option_cost_rate / 2.0
        if self.cost_model is None:
            return 0.0
        return notional * float(self.cost_model.per_leg_bps(notional, side)) / 10_000.0

    def _atr_of(self, state: LegState) -> float:
        try:
            return max(float(state.atr or 0.0), 0.0)
        except (TypeError, ValueError):
            return 0.0

    # ------------------------------------------------------------- bookkeeping

    def _mark(self, prices: Mapping[str, float]) -> None:
        for leg, price in prices.items():
            position = self.positions.get(leg)
            unrealized = position.unrealized(float(price or 0.0)) if position else 0.0
            self._wallet(leg).observe(unrealized)

    def _persist(self) -> None:
        if self.ledger is None:
            return
        try:
            self.ledger.save_wallets(self.wallets, self.reserve)
        except Exception as exc:  # noqa: BLE001 — never lose the session to the book
            self.log.warning("failed to persist the money book: %s", exc)

    # -------------------------------------------------------------- snapshot

    @property
    def deployed_capital(self) -> float:
        """Cash sitting in the legs, excluding the reserve."""
        return sum(wallet.cash for wallet in self.wallets.values())

    def total_pnl(self) -> float:
        return sum(wallet.total_pnl for wallet in self.wallets.values())

    def headline(self) -> str:
        if not self.wallets:
            return "no capital deployed yet"
        pnl = self.total_pnl()
        best = max(self.wallets.values(), key=lambda wallet: wallet.total_pnl)
        return (
            f"net {pnl:+,.0f} across {len(self.wallets)} legs · "
            f"best {best.label} {best.total_pnl:+,.0f} · reserve {self.reserve.remaining:,.0f} left"
        )

    def snapshot(self, now: datetime | None = None) -> dict:
        """Everything the dashboard needs, in rupees."""
        with self._lock:
            moment = now or self.updated_at or datetime.now()
            legs = sorted(set(self.wallets) | set(self.positions) | set(self.views))
            wallets = []
            for leg in legs:
                wallet = self._wallet(leg)
                position = self.positions.get(leg)
                view = self.views.get(leg)
                price = view.price if view is not None else 0.0
                unrealized = position.unrealized(price) if position and price > 0 else 0.0
                row = wallet.as_row(unrealized)
                row["position"] = position.as_row(price, moment) if position else None
                row["view"] = view.as_row() if view is not None else None
                wallets.append(row)

            return {
                "headline": self.headline(),
                "opening_balance": self.config.opening_balance,
                "lots": self.config.lot_size,
                "reserve": self.reserve.as_row(),
                "capital_committed": self.config.opening_balance * len(wallets) + self.config.reserve,
                "cash": round(self.deployed_capital, 2),
                "unrealized": round(sum(item["unrealized"] for item in wallets), 2),
                "equity": round(sum(item["equity"] for item in wallets), 2),
                "net_pnl": round(self.total_pnl(), 2),
                "wallets": wallets,
                "decisions": [decision.as_row() for decision in reversed(self.decisions)],
                "trades": [trade.as_row() for trade in reversed(self.closed)],
                "policy": self.config.as_row(),
                "steps": self.steps,
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "updated_at": moment.isoformat(),
            }

    def __repr__(self) -> str:
        return (
            f"<MoneySimulator {len(self.wallets)} wallets, "
            f"{len(self.positions)} open, {self.total_pnl():+,.0f} net>"
        )


__all__ = ["DEFAULT_MIN_EQUITY_FRACTION", "MoneySimulator", "SimulatorConfig"]
