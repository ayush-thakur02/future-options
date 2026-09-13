"""The call/index/put board: several instruments driven by one tick stream.

The index has a feed of its own. A leg does **not** — its premium is a function of
the index price, the strike, the time left and the implied vol. So deriving a leg
tick from an index tick is not an approximation standing in for a missing feed; it
is the definition of what a leg is worth. The practical consequence is that the
three charts cannot disagree about what the market did, which is the failure mode
a multi-instrument display has to avoid above all others.

Each instrument gets its own :class:`~runtime.engine.Engine` — its own bars, its
own features, its own strategies and its own projected path — and all of them are
refreshed by one clock. A leg engine is an engine: nothing about it is special
because its prices happen to be premiums.

Legs are re-struck when the spot walks away from them. A roll is a new instrument,
so the leg's history is regenerated from the index history at the new strike and
its projection scorecard is reset — scoring a projection made about one strike
against the bars of another would be a quiet lie.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from core.calendar import IST
from core.settings import Settings
from core.types import BoardSnapshot, LegSnapshot, Tick
from kernel import Kernel

from .engine import Engine

INDEX_LABEL = "INDEX"
CALL_KIND = "CE"
PUT_KIND = "PE"
# How far the spot may drift from a leg's strike before the leg is re-struck.
ROLL_STRIKES = 2.0


@dataclass
class BoardLeg:
    """One instrument on the board, and the engine running it."""

    label: str
    kind: str
    engine: Engine
    strike: float = 0.0
    step: float = 50.0
    greeks: dict = field(default_factory=dict)
    rolls: int = 0

    @property
    def is_option(self) -> bool:
        return self.kind in {CALL_KIND, PUT_KIND}

    @property
    def atm_iv(self) -> float:
        return float(self.greeks.get("iv", 0.0))

    def summary(self) -> str:
        if not self.is_option:
            return f"{self.label} {self.engine.last_price:,.2f}"
        return (
            f"{self.label} {self.strike:,.0f} @ {self.engine.last_price:,.2f} "
            f"Δ{self.greeks.get('delta', 0.0):+.2f}"
        )


class MarketBoard:
    """The index and its two at-the-money legs, on one clock."""

    def __init__(
        self,
        kernel: Kernel,
        bar_minutes: int | None = None,
        roll_strikes: float = ROLL_STRIKES,
        logger: logging.Logger | None = None,
    ) -> None:
        self.kernel = kernel
        self.settings: Settings = kernel.settings
        self.bar_minutes = int(bar_minutes or kernel.settings.bar_minutes)
        self.roll_strikes = float(roll_strikes)
        self.log = logger or logging.getLogger("niftypulse.board")

        self.source = kernel.capability("option_chain") if kernel.registry.capabilities().get("option_chain") else None
        self.advisor = kernel.capability("advisory") if kernel.registry.capabilities().get("advisory") else None
        self.legs: list[BoardLeg] = []
        self.index_bars: pd.DataFrame = pd.DataFrame()
        self.expiry = None
        self.ticks = 0
        # Called at the end of every refresh, so a session can sample the chain on
        # the same clock the projections run on rather than inventing a second one.
        self.on_refresh: Callable[[], None] | None = None

    # ---------------------------------------------------------------- assembly

    @property
    def index_engine(self) -> Engine:
        return self.legs[0].engine

    def leg(self, label: str) -> BoardLeg | None:
        for item in self.legs:
            if item.label == label:
                return item
        return None

    def build(self, index_bars: pd.DataFrame) -> MarketBoard:
        """Warm every instrument up, so the first frame is already readable."""
        self.index_bars = index_bars
        index_engine = Engine(
            self.kernel,
            bar_minutes=self.bar_minutes,
            symbol=self.settings.symbol,
        ).bootstrap(history=index_bars)
        self.legs = [BoardLeg(label=INDEX_LABEL, kind="index", engine=index_engine)]

        if self.source is None:
            self.log.info("no option chain capability; the board will show the index alone")
            return self

        spot = index_engine.last_price
        self.expiry = self.source.expiry()
        for kind, label in ((CALL_KIND, "CALL"), (PUT_KIND, "PUT")):
            leg = self._make_leg(kind, label, index_bars, spot)
            if leg is not None:
                self.legs.append(leg)

        # Leave every instrument with a path ready to draw: a board that opened on
        # three empty charts until the first tick arrived would make the first
        # second of every session look broken.
        self.refresh_projection()
        return self

    def _make_leg(self, kind: str, label: str, index_bars: pd.DataFrame, spot: float) -> BoardLeg | None:
        chain_legs = self.source.legs(index_bars, spot=spot)
        chain_leg = chain_legs.get(kind)
        if chain_leg is None or chain_leg.bars.empty:
            return None

        engine = Engine(
            self.kernel,
            bar_minutes=self.bar_minutes,
            symbol=f"{self.settings.symbol} {chain_leg.strike:,.0f} {kind}",
        ).bootstrap(history=chain_leg.bars)

        return BoardLeg(
            label=label,
            kind=kind,
            engine=engine,
            strike=chain_leg.strike,
            step=float(self.source.strike_step),
            greeks=self._greeks(chain_leg),
        )

    def _greeks(self, chain_leg) -> dict:
        """Leg greeks, plus the two contextual numbers the verdict needs.

        Both come from the source rather than being recomputed here: how far apart
        strikes sit and what the underlying has actually been realising are
        properties of the chain and its index, not of the board drawing it.
        """
        greeks = dict(chain_leg.greeks)
        greeks["realised_vol"] = self.source.realised_vol
        greeks["strike_step"] = float(self.source.strike_step)
        return greeks

    # ------------------------------------------------------------------ ticks

    def on_tick(self, tick: Tick) -> bool:
        """Feed one index tick, and the leg ticks it implies."""
        self.ticks += 1
        closed = self.index_engine.on_tick(tick)
        spot = tick.ltp or tick.mid

        for leg in self.legs[1:]:
            derivative = self._leg_tick(leg, spot, tick)
            if derivative is not None:
                leg.engine.on_tick(derivative)

        self.roll_if_needed(spot)
        return closed

    def _leg_tick(self, leg: BoardLeg, spot: float, index_tick: Tick) -> Tick | None:
        """The premium tick a leg would have printed at this index price."""
        if self.source is None or spot <= 0:
            return None
        moment = index_tick.ts if index_tick.ts.tzinfo else index_tick.ts.replace(tzinfo=IST)
        quote = self.source.quote(
            leg.kind,
            leg.strike,
            spot,
            moment,
            atm_iv=list(self.legs[1:])[0].atm_iv or None,
        )
        leg.greeks = {**leg.greeks, **self._quote_greeks(quote)}
        if quote.premium <= 0:
            return None
        return Tick(ts=index_tick.ts, ltp=quote.premium, ltq=index_tick.ltq, close_prev=index_tick.close_prev)

    def _quote_greeks(self, quote) -> dict:
        return {
            "premium": quote.premium,
            "delta": quote.delta,
            "gamma": quote.gamma,
            "theta_per_minute": quote.theta_per_minute,
            "vega": quote.vega,
            "iv": quote.iv,
            "minutes_to_expiry": quote.minutes_to_expiry,
        }

    # ------------------------------------------------------------------- rolls

    def roll_if_needed(self, spot: float) -> list[str]:
        """Re-strike any leg the spot has walked away from.

        A leg two strikes from the money is no longer the trade the board is
        about — it is a directional bet with a small delta. Rolling keeps both
        legs at the money, which is what makes the call and the put comparable
        and what keeps the panel's greeks meaningful.
        """
        if self.source is None or spot <= 0:
            return []

        target = self.source.atm_strike(spot)
        rolled: list[str] = []
        for leg in self.legs[1:]:
            if abs(spot - leg.strike) < self.roll_strikes * leg.step:
                continue

            chain_leg = self.source.legs(self.index_bars, spot=spot).get(leg.kind)
            if chain_leg is None or chain_leg.bars.empty:
                continue

            leg.engine.history = pd.DataFrame()
            leg.engine.bootstrap(history=chain_leg.bars)
            # A roll is a different instrument: a projection made about the old
            # strike must not be scored against the new one's bars.
            if leg.engine.forecaster is not None:
                leg.engine.forecaster.tracker.reset()
                leg.engine.forecaster.momentum.reset()
            leg.engine.projections = []
            leg.strike = chain_leg.strike
            leg.greeks = self._greeks(chain_leg)
            leg.rolls += 1
            rolled.append(f"{leg.label} → {leg.strike:,.0f}")

        if rolled:
            self.log.info("rolled legs to %s (atm %s)", ", ".join(rolled), f"{target:,.0f}")
        return rolled

    # -------------------------------------------------------------- projection

    def refresh_projection(self) -> bool:
        """Rebuild every instrument's path on the shared clock."""
        produced = self.index_engine.refresh_projection()
        for leg in self.legs[1:]:
            leg.engine.refresh_projection()
        if self.on_refresh is not None:
            self.on_refresh()
        return produced

    def publish(self) -> None:
        for leg in self.legs:
            leg.engine.publish()

    # --------------------------------------------------------------- snapshot

    def snapshot(self) -> BoardSnapshot:
        spot_engine = self.index_engine
        board = BoardSnapshot(
            ts=spot_engine.snapshot().ts,
            symbol=self.settings.symbol,
            spot=spot_engine.last_price,
            legs=[
                LegSnapshot(
                    label=leg.label,
                    kind=leg.kind,
                    snapshot=leg.engine.snapshot(),
                    strike=leg.strike,
                    greeks=dict(leg.greeks),
                )
                for leg in self.legs
            ],
            chain=self._chain_summary(),
        )
        if self.advisor is not None:
            board = self.advisor.advise(board, bar_minutes=self.bar_minutes)
        return board

    def _chain_summary(self) -> dict:
        if self.source is None or self.index_bars is None or self.index_bars.empty:
            return {}
        chain = self.source.chain(self.index_engine.last_price, bars=self.index_bars)
        totals = chain.totals()
        totals["expiry"] = chain.expiry.strftime("%a %d %b")
        totals["strikes"] = len(chain.strikes())
        return totals

    # ------------------------------------------------------------ diagnostics

    def describe(self) -> str:
        parts = [f"{len(self.legs)} legs"]
        for leg in self.legs[1:]:
            parts.append(f"{leg.kind} {leg.strike:,.0f}")
        if self.expiry is not None:
            parts.append(f"expiry {self.expiry:%a %d %b}")
        return " · ".join(parts)

    def __repr__(self) -> str:
        return f"<MarketBoard {self.describe()} after {self.ticks} ticks>"


__all__ = ["BoardLeg", "MarketBoard", "ROLL_STRIKES"]
