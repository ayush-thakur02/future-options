"""Independent index/call/put engines. Live options use exchange ticks;
calculated premiums are restricted to the explicitly offline simulation.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import pandas as pd

from core.calendar import IST
from core.settings import Settings
from core.types import BoardSnapshot, LegSnapshot, Tick
from kernel import Kernel

from .engine import FEATURE_WINDOW, Engine

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
    instrument_key: str = ""

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
        history_bars: int = FEATURE_WINDOW,
        logger: logging.Logger | None = None,
        live: bool = False,
        days: int = 15,
    ) -> None:
        self.kernel = kernel
        self.settings: Settings = kernel.settings
        self.bar_minutes = int(bar_minutes or kernel.settings.bar_minutes)
        self.roll_strikes = float(roll_strikes)
        # How much history each instrument carries. The engines discard anything
        # beyond their feature window anyway, and the charts show a fraction of
        # that, so pricing premium candles across a multi-year cache would be work
        # thrown away — three engines' worth of it, on every start and every roll.
        self.history_bars = int(history_bars)
        self.log = logger or logging.getLogger("niftypulse.board")
        self.live = live
        self.executor = ThreadPoolExecutor(max_workers=self.settings.worker_count, thread_name_prefix="instrument")

        self.source = kernel.capability("option_chain") if kernel.registry.capabilities().get("option_chain") else None
        if live:
            from plugins.sources.upstox.options import LiveOptionSource

            self.source = LiveOptionSource(kernel.build("source:upstox"), self.executor, days)
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
        index_bars = index_bars.tail(self.history_bars) if index_bars is not None else index_bars
        self.index_bars = index_bars
        index_engine = Engine(
            self.kernel,
            bar_minutes=self.bar_minutes,
            symbol=self.settings.symbol,
            instrument_key=self.settings.instrument_key,
            source="Upstox" if self.live else "simulation",
        ).bootstrap(history=index_bars)
        self.legs = [BoardLeg(label=INDEX_LABEL, kind="index", engine=index_engine,
                              instrument_key=self.settings.instrument_key)]

        if self.live:
            selected = self.source.select(index_bars)
            self.expiry = self.source.expiry()
            def warm(item):
                kind, contract = item
                engine = Engine(self.kernel, bar_minutes=self.bar_minutes, symbol=contract.symbol,
                                instrument_key=contract.instrument_key, source="Upstox",
                                anchor_provider=index_engine.aligned_close)
                engine.bootstrap(history=contract.bars)
                return BoardLeg(label="CALL" if kind == "CE" else "PUT", kind=kind,
                                engine=engine, strike=contract.strike, step=self.source.strike_step,
                                greeks=contract.greeks, instrument_key=contract.instrument_key)
            self.legs.extend(self.executor.map(warm, sorted(selected.items())))
            self.refresh_projection()
            return self

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
            instrument_key=f"SIM|{self.settings.symbol}:{chain_leg.strike}:{kind}:{self.expiry}",
            anchor_provider=self.index_engine.aligned_close,
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
        if self.live:
            for leg in self.legs:
                if leg.instrument_key == tick.instrument_key:
                    if tick.greeks:
                        leg.greeks.update({k: v for k, v in tick.greeks.items() if k not in {"iv", "theta"}})
                        leg.greeks["iv"] = tick.greeks.get("iv", 0) / 100.0
                        leg.greeks["theta_per_minute"] = tick.greeks.get("theta", 0) / 1440.0
                    leg.greeks["premium"] = tick.ltp
                    return leg.engine.on_tick(tick)
            return False
        closed = self.index_engine.on_tick(tick)
        self.index_bars = self.index_engine.history
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
        return Tick(ts=index_tick.ts, ltp=quote.premium, ltq=index_tick.ltq, close_prev=leg.engine.prev_close)

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
        if self.live or self.source is None or spot <= 0:
            return []

        target = self.source.atm_strike(spot)
        rolled: list[str] = []
        for leg in self.legs[1:]:
            if abs(spot - leg.strike) < self.roll_strikes * leg.step:
                continue

            replacement = self._make_leg(leg.kind, leg.label, self.index_bars, spot)
            if replacement is None:
                continue
            research = leg.engine.forecaster and leg.engine.forecaster.tracker.journal is not None
            leg.engine.close()
            leg.engine = replacement.engine
            if research:
                leg.engine.enable_research()
            leg.strike = replacement.strike
            leg.greeks = replacement.greeks
            leg.rolls += 1
            rolled.append(f"{leg.label} → {leg.strike:,.0f}")

        if rolled:
            self.log.info("rolled legs to %s (atm %s)", ", ".join(rolled), f"{target:,.0f}")
        return rolled

    # -------------------------------------------------------------- projection

    def refresh_projection(self) -> bool:
        """Rebuild every instrument's path on the shared clock."""
        produced = list(self.executor.map(lambda leg: leg.engine.refresh_projection(), self.legs))
        if self.on_refresh is not None:
            self.on_refresh()
        return any(produced)

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)
        for leg in self.legs:
            leg.engine.close()

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
