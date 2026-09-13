"""The index option chain as a plugin.

Provides the ``option_chain`` capability: expiries, the strikes around the money,
per-strike greeks and open interest, and the two at-the-money legs the board
charts.

Everything here is priced from the index series the rest of the platform is
already looking at, so a leg's candles, its greeks and its breakeven requirement
are consistent with the chart beside them. The live chain — the same shape, read
from the account instead of computed — is a matter of adding a reader module and
declaring the ``broker`` requirement; the leg interface below does not change.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from core.calendar import IST
from core.settings import Settings
from kernel import PluginContext, PluginKind, PluginManifest

from .chain import (
    CALL,
    PUT,
    STRIKE_STEP,
    LegQuote,
    OptionChain,
    chain_iv,
    leg_bars,
    next_weekly_expiry,
    smile_iv,
    synthetic_chain,
    trading_minutes_to,
)
from .pricing import (
    bs_delta,
    bs_gamma,
    bs_price,
    bs_theta_per_minute,
    bs_vega_per_vol_point,
    years_from_minutes,
)

MANIFEST = PluginManifest(
    name="option_chain",
    kind=PluginKind.SOURCE,
    description="Index option chain: expiries, strikes, greeks, open interest",
    provides=("option_chain",),
    tags=("options", "derivatives", "chain"),
    params={"width": 10, "expiry_weekday": 1},
)

DEFAULT_WIDTH = 10


@dataclass(slots=True)
class ChainLeg:
    """One side of the money, as the board charts it."""

    kind: str
    strike: float
    bars: pd.DataFrame
    quote: LegQuote

    @property
    def label(self) -> str:
        return "CALL" if self.kind == CALL else "PUT"

    @property
    def greeks(self) -> dict:
        return {
            "premium": self.quote.premium,
            "delta": self.quote.delta,
            "gamma": self.quote.gamma,
            "theta_per_minute": self.quote.theta_per_minute,
            "vega": self.quote.vega,
            "iv": self.quote.iv,
            "oi": self.quote.oi,
            "strike": self.strike,
            "breakeven_bps": self.quote.breakeven_move_bps(),
            "minutes_to_expiry": self.quote.minutes_to_expiry,
        }


class OptionChainSource:
    """The chain, and the at-the-money legs derived from it."""

    def __init__(
        self,
        settings: Settings,
        width: int = DEFAULT_WIDTH,
        expiry_weekday: int = 1,
    ) -> None:
        self.settings = settings
        self.width = int(width)
        self.expiry_weekday = int(expiry_weekday)
        self._expiry: datetime | None = None

    # ------------------------------------------------------------------ expiry

    def expiry(self, moment: datetime | None = None) -> datetime:
        """The expiry the board is trading, cached until it passes."""
        moment = (moment or datetime.now(IST)).astimezone(IST)
        if self._expiry is None or self._expiry <= moment:
            self._expiry = next_weekly_expiry(moment, self.expiry_weekday)
        return self._expiry

    def minutes_to_expiry(self, moment: datetime | None = None) -> float:
        moment = (moment or datetime.now(IST)).astimezone(IST)
        return trading_minutes_to(moment, self.expiry(moment))

    def atm_strike(self, spot: float) -> float:
        return float(round(spot / STRIKE_STEP) * STRIKE_STEP)

    # ------------------------------------------------------------------- chain

    def iv(self, bars: pd.DataFrame) -> float:
        """The vol a synthetic chain carries, from the bars it is built on."""
        return chain_iv(bars["close"]) if bars is not None and not bars.empty else 0.12

    def chain(
        self,
        spot: float,
        moment: datetime | None = None,
        bars: pd.DataFrame | None = None,
    ) -> OptionChain:
        moment = (moment or datetime.now(IST)).astimezone(IST)
        return synthetic_chain(
            spot=float(spot),
            ts=moment,
            expiry=self.expiry(moment),
            atm_iv=self.iv(bars) if bars is not None else None,
            width=self.width,
        )

    def quote(
        self,
        kind: str,
        strike: float,
        spot: float,
        moment: datetime | None = None,
        atm_iv: float | None = None,
    ) -> LegQuote:
        """One strike, one side, at one moment — the cheap path a tick can call."""
        moment = (moment or datetime.now(IST)).astimezone(IST)
        minutes = trading_minutes_to(moment, self.expiry(moment))
        years = years_from_minutes(minutes)
        moneyness = ((strike - spot) / spot) * 10_000.0 if spot else 0.0
        iv = smile_iv(atm_iv if atm_iv else 0.12, moneyness)
        return LegQuote(
            strike=float(strike),
            kind=kind,
            premium=bs_price(spot, strike, iv, years, kind),
            delta=bs_delta(spot, strike, iv, years, kind),
            gamma=bs_gamma(spot, strike, iv, years),
            theta_per_minute=bs_theta_per_minute(spot, strike, iv, years),
            vega=bs_vega_per_vol_point(spot, strike, iv, years),
            iv=iv,
            oi=0.0,
            volume=0.0,
            spot=float(spot),
            minutes_to_expiry=minutes,
        )

    # -------------------------------------------------------------------- legs

    def legs(
        self,
        index_bars: pd.DataFrame,
        spot: float | None = None,
        moment: datetime | None = None,
    ) -> dict[str, ChainLeg]:
        """The at-the-money call and put, with premium candles and greeks.

        Both legs share a strike when the spot sits near a round number and take
        the two nearest strikes when it does not — which is what a straddle
        seller's board looks like, and it keeps the call and the put comparable
        rather than silently one strike apart.
        """
        if index_bars is None or index_bars.empty:
            return {}

        moment = (moment or index_bars.index[-1].to_pydatetime()).astimezone(IST)
        spot = float(spot if spot is not None else index_bars["close"].iloc[-1])
        expiry = self.expiry(moment)
        atm = self.atm_strike(spot)
        atm_iv = self.iv(index_bars)
        chain = self.chain(spot, moment, index_bars)

        legs: dict[str, ChainLeg] = {}
        for kind in (CALL, PUT):
            strike = atm
            quote = chain.at(strike, kind) or chain.atm(kind)
            if quote is None:
                continue
            bars = leg_bars(index_bars, strike, kind, expiry, atm_iv=atm_iv)
            if bars.empty:
                continue
            legs[kind] = ChainLeg(kind=kind, strike=strike, bars=bars, quote=quote)
        return legs


def build(ctx: PluginContext, width: int = DEFAULT_WIDTH, expiry_weekday: int | None = None, **params) -> OptionChainSource:
    return OptionChainSource(
        settings=ctx.settings,
        width=width,
        expiry_weekday=ctx.settings.expiry_weekday if expiry_weekday is None else int(expiry_weekday),
    )


__all__ = ["MANIFEST", "ChainLeg", "OptionChainSource", "build"]
