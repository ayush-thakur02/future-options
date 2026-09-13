"""The option chain, and a synthetic one for offline work.

Two things live here that the rest of the platform needs:

* :class:`LegQuote` — one strike, one side, at one moment: premium, greeks, IV,
  open interest, and the breakeven move it implies.
* :class:`OptionChain` — the strikes around the money, plus the aggregate
  positioning a reader actually uses (put/call open interest, PCR).

The synthetic chain exists so the call/index/put board can be built and tested
without an account. It is not a random premium series: it is priced with
:mod:`~plugins.sources.option_chain.pricing` off the real index bars, which means
the leg's candles, its greeks and its breakeven requirement are all internally
consistent with the chart next to them, and the theta decay across a session is
the real shape rather than a drift term someone chose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from math import exp

import numpy as np
import pandas as pd

from core.calendar import IST

from .pricing import (
    CALL,
    PUT,
    breakeven_move_bps,
    bs_delta,
    bs_gamma,
    bs_price,
    bs_theta_per_minute,
    bs_vega_per_vol_point,
    years_from_minutes,
)

STRIKE_STEP = 50
CHAIN_STEP = 3

# Floor on the implied vol used by the *synthetic* chain only. Realised vol on
# the bundled generated series annualises to around 3%, which is far below
# anything NIFTY options have ever traded at, and a chain priced off it would
# show premiums an order of magnitude too cheap for the panel to be read as a
# real one. Live chains are never touched by this.
SYNTHETIC_IV_FLOOR = 0.10
# The market prices a premium to what the underlying has actually been doing.
SYNTHETIC_IV_PREMIUM = 1.15
SESSION_OPEN_TIME = time(9, 15)
SESSION_CLOSE_TIME = time(15, 30)
SESSION_MINUTES_PER_DAY = 375


# --------------------------------------------------------------------- clock


def trading_minutes_to(moment: datetime, expiry: datetime) -> float:
    """Trading minutes between two instants, counting only session time.

    An option does not decay overnight or over a weekend, so a chain priced off
    the wall clock would show Friday's theta spread across a week of closed
    hours. Weekends are skipped explicitly; exchange holidays are not, because a
    wrong holiday would be worse than a slightly generous decay.
    """
    moment = moment.astimezone(IST)
    expiry = expiry.astimezone(IST)
    if expiry <= moment:
        return 0.0

    total = 0.0
    day = moment.date()
    while day <= expiry.date():
        # Weekends are skipped: the exchange is shut, so no decay accrues. Holiday
        # gaps are not modelled, which overstates decay across a week containing
        # one — the calendar module holds the holiday list if that ever matters.
        if day.weekday() < 5:
            open_at = datetime.combine(day, SESSION_OPEN_TIME, tzinfo=IST)
            close_at = datetime.combine(day, SESSION_CLOSE_TIME, tzinfo=IST)
            start = max(moment, open_at)
            end = min(expiry, close_at)
            if end > start:
                total += (end - start).total_seconds() / 60.0
        day += timedelta(days=1)
    return total


def next_weekly_expiry(moment: datetime, weekday: int = 1) -> datetime:
    """The coming weekly expiry at the session close, on ``weekday`` (0=Monday)."""
    day = moment.astimezone(IST).date()
    ahead = (weekday - day.weekday()) % 7
    target = day + timedelta(days=ahead)
    expiry = datetime.combine(target, SESSION_CLOSE_TIME, tzinfo=IST)
    if expiry <= moment:
        expiry += timedelta(days=7)
    return expiry


# ---------------------------------------------------------------------- legs


@dataclass(slots=True)
class LegQuote:
    """One strike, one side, at one moment."""

    strike: float
    kind: str
    premium: float
    delta: float
    gamma: float
    theta_per_minute: float
    vega: float
    iv: float
    oi: float
    volume: float
    spot: float
    minutes_to_expiry: float

    @property
    def is_call(self) -> bool:
        return self.kind == CALL

    @property
    def moneyness(self) -> float:
        """Strike less spot, in basis points of spot. Positive is out of the money."""
        if self.spot <= 0:
            return 0.0
        return ((self.strike - self.spot) / self.spot) * 10_000.0

    @property
    def money_per_minute(self) -> float:
        """Rupees of premium lost per minute of holding, as a positive cost."""
        return abs(self.theta_per_minute)

    def breakeven_move_bps(self, extra_cost: float = 0.0) -> float:
        """Underlying move needed for this leg to pay for itself.

        ``extra_cost`` lets a caller fold in brokerage and slippage, which on a
        scalping horizon is not a rounding error.
        """
        return breakeven_move_bps(self.premium + extra_cost, abs(self.delta), self.spot)

    def as_row(self) -> dict:
        return {
            "strike": self.strike,
            "kind": self.kind,
            "premium": round(self.premium, 2),
            "delta": round(self.delta, 4),
            "theta_per_minute": round(self.theta_per_minute, 4),
            "iv": round(self.iv, 4),
            "oi": self.oi,
            "breakeven_bps": round(self.breakeven_move_bps(), 1),
        }


@dataclass(slots=True)
class OptionChain:
    """The strikes around the money, at one moment."""

    ts: datetime
    spot: float
    expiry: datetime
    minutes_to_expiry: float
    strike_step: int
    atm_strike: float
    quotes: dict[tuple[float, str], LegQuote] = field(default_factory=dict)

    def at(self, strike: float, kind: str) -> LegQuote | None:
        return self.quotes.get((float(strike), kind))

    def atm(self, kind: str) -> LegQuote | None:
        return self.at(self.atm_strike, kind)

    def strikes(self) -> list[float]:
        return sorted({strike for strike, _ in self.quotes})

    def side(self, kind: str) -> list[LegQuote]:
        return [quote for (_, quote_kind), quote in sorted(self.quotes.items()) if quote_kind == kind]

    def totals(self) -> dict:
        """Aggregate positioning, which is what a chain is read for."""
        call_oi = sum(quote.oi for quote in self.side(CALL))
        put_oi = sum(quote.oi for quote in self.side(PUT))
        return {
            "call_oi": call_oi,
            "put_oi": put_oi,
            "pcr": (put_oi / call_oi) if call_oi else 0.0,
            "spot": self.spot,
            "atm_strike": self.atm_strike,
            "minutes_to_expiry": round(self.minutes_to_expiry, 1),
        }

    def describe(self) -> str:
        totals = self.totals()
        return (
            f"{len(self.strikes())} strikes around {self.atm_strike:,.0f}, "
            f"PCR {totals['pcr']:.2f}, {totals['minutes_to_expiry']:.0f} min to expiry"
        )


# ------------------------------------------------------------- volume & vol


def annualised_vol(closes: pd.Series, window: int = 30) -> float:
    """Realised volatility from minute bars, annualised over trading minutes."""
    if closes is None or len(closes) < 3:
        return 0.12
    returns = np.diff(np.log(closes.astype("float64").to_numpy()[-window - 1 :]))
    if returns.size == 0:
        return 0.12
    per_minute = float(np.std(returns, ddof=1))
    annual = per_minute * np.sqrt(SESSION_MINUTES_PER_DAY * 252)
    return float(np.clip(annual, 0.02, 1.5))


def chain_iv(closes: pd.Series, window: int = 30) -> float:
    """The implied vol a synthetic chain should carry, from the index it tracks."""
    return float(max(annualised_vol(closes, window) * SYNTHETIC_IV_PREMIUM, SYNTHETIC_IV_FLOOR))


def smile_iv(atm_iv: float, moneyness_bps: float, curvature: float = 0.06) -> float:
    """Implied vol as a function of how far the strike sits from spot.

    A parabola in moneyness: the wings trade richer than the middle, which is
    what an equity index smile looks like. Without it every strike would carry
    the same vol and the chain would have no reason to prefer one strike over
    another — a preference the board is supposed to reveal.
    """
    scaled = moneyness_bps / 100.0
    return float(max(atm_iv * (1.0 + curvature * scaled * scaled), 0.01))


# ----------------------------------------------------------------- synthetic


def synthetic_chain(
    spot: float,
    ts: datetime,
    expiry: datetime | None = None,
    atm_iv: float | None = None,
    width: int = 10,
    strike_step: int = STRIKE_STEP,
    seed: int = 7,
) -> OptionChain:
    """A plausible chain around the money, priced off ``spot``.

    Open interest is shaped rather than random: it peaks at the round strikes,
    thins with distance, and leans — calls above spot, puts below — which is the
    structure that makes PCR and max-pain style readings mean anything at all.
    """
    moment = ts.astimezone(IST)
    expiry = expiry or next_weekly_expiry(moment)
    minutes = trading_minutes_to(moment, expiry)
    years = years_from_minutes(minutes)
    atm_strike = round(spot / strike_step) * strike_step
    vol = float(atm_iv) if atm_iv else 0.12
    rng = np.random.default_rng(seed + int(spot) % 1000)

    chain = OptionChain(
        ts=moment,
        spot=float(spot),
        expiry=expiry,
        minutes_to_expiry=minutes,
        strike_step=strike_step,
        atm_strike=float(atm_strike),
    )

    for offset in range(-width, width + 1):
        strike = float(atm_strike + offset * strike_step)
        if strike <= 0:
            continue
        moneyness = ((strike - spot) / spot) * 10_000.0
        iv = smile_iv(vol, moneyness)
        for kind in (CALL, PUT):
            premium = bs_price(spot, strike, iv, years, kind)
            chain.quotes[(strike, kind)] = LegQuote(
                strike=strike,
                kind=kind,
                premium=premium,
                delta=bs_delta(spot, strike, iv, years, kind),
                gamma=bs_gamma(spot, strike, iv, years),
                theta_per_minute=bs_theta_per_minute(spot, strike, iv, years),
                vega=bs_vega_per_vol_point(spot, strike, iv, years),
                iv=iv,
                oi=_synthetic_oi(offset, width, kind, rng),
                volume=0.0,
                spot=float(spot),
                minutes_to_expiry=minutes,
            )
    return chain


def seed_of(strike: float, kind: str) -> int:
    """A stable seed for one strike and side.

    Not ``hash()``: Python salts string hashes per process, so a "reproducible"
    synthetic chain would come out different on every run and a bug found in one
    would not be findable in the next.
    """
    kind_code = 1 if kind == CALL else 2
    return int(strike) * 17 + kind_code * 7919


def rng_unit(strike: float, step: int) -> float:
    """A deterministic pseudo-random value in [0, 1) for one strike and bar."""
    rng = np.random.default_rng(seed_of(strike, CALL) + step * 104_729)
    return float(rng.random())


def _synthetic_oi(offset: int, width: int, kind: str, rng: np.random.Generator) -> float:
    """Open interest for one strike: peaked at the money, leaning with the side.

    Calls build up above spot and puts below it, because that is where writers
    and hedgers actually sit. A symmetric blob would make the put/call ratio
    meaningless and the chain panel decorative.
    """
    lean = 0.75 if (kind == CALL and offset >= 0) or (kind == PUT and offset <= 0) else 0.35
    round_strike = 1.25 if offset % 2 == 0 else 1.0
    falloff = exp(-0.5 * (offset / max(width * 0.45, 1)) ** 2)
    noise = float(rng.uniform(0.85, 1.15))
    return float(round(60_000 * lean * round_strike * falloff * noise, 0))


def leg_bars(
    index_bars: pd.DataFrame,
    strike: float,
    kind: str,
    expiry: datetime,
    atm_iv: float | None = None,
    bar_minutes: int = 1,
) -> pd.DataFrame:
    """Premium candles for one leg, priced off the index bars beside them.

    Each index bar's open, high, low and close are re-priced at the same implied
    vol, so the premium candle is a monotone transform of the index candle and
    the two charts cannot tell different stories. The greeks are attached as
    columns and read at the bar's close, which is what the panel reports.

    Volume is synthesised from open interest rather than left at zero: the index
    has no volume to inherit, and a leg chart with a flat volume column would
    silently disable every activity feature built on top of it.
    """
    if index_bars is None or index_bars.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "oi"])

    vol_floor = chain_iv(index_bars["close"]) if atm_iv is None else float(atm_iv)
    strike = float(strike)
    rows: list[dict] = []
    index: list[pd.Timestamp] = []

    # Open interest is anchored to the strike grid as it stands at the start of
    # the series, not re-derived per bar: an OI column that wandered with spot
    # would make every "positioning changed" reading an artifact of the anchor.
    anchor_spot = float(index_bars["close"].iloc[0])
    anchor_atm = round(anchor_spot / STRIKE_STEP) * STRIKE_STEP
    offset = (strike - anchor_atm) / STRIKE_STEP
    base_oi = _synthetic_oi(int(offset), CHAIN_STEP * 3, kind, np.random.default_rng(seed_of(strike, kind)))

    for stamp, bar in index_bars.iterrows():
        moment = stamp.to_pydatetime() if hasattr(stamp, "to_pydatetime") else stamp
        minutes = trading_minutes_to(moment, expiry)
        years = years_from_minutes(minutes)
        moneyness = ((strike - float(bar["close"])) / float(bar["close"])) * 10_000.0
        iv = smile_iv(vol_floor, moneyness)

        opening = bs_price(float(bar["open"]), strike, iv, years, kind)
        high = bs_price(float(bar["high"]), strike, iv, years, kind)
        low = bs_price(float(bar["low"]), strike, iv, years, kind)
        close = bs_price(float(bar["close"]), strike, iv, years, kind)
        # A premium is monotone in spot for a call and decreasing for a put, so
        # repricing the extremes the other way round would produce candles that
        # contradict the index chart. Enforce it rather than trust the ordering.
        high, low = max(high, low, opening, close), min(high, low, opening, close)

        # A slow wobble plus a size proportional to how far the bar travelled:
        # this is a proxy, and the activity features that consume it only need it
        # to be busy when the market is busy.
        travelled = abs(close - opening) / max(opening, 1e-9)
        oi = base_oi * (1.0 + 0.06 * float(np.sin(len(index) / 9.0)) + 0.02 * float(rng_unit(strike, len(index))))
        volume = base_oi * 0.15 * (1.0 + 40.0 * travelled)

        index.append(stamp)
        rows.append(
            {
                "open": opening,
                "high": high,
                "low": low,
                "close": close,
                "volume": float(volume),
                "oi": float(oi),
                "spot": float(bar["close"]),
                "delta": bs_delta(float(bar["close"]), strike, iv, years, kind),
                "gamma": bs_gamma(float(bar["close"]), strike, iv, years),
                "theta_per_minute": bs_theta_per_minute(float(bar["close"]), strike, iv, years),
                "iv": iv,
                "minutes_to_expiry": minutes,
            }
        )

    frame = pd.DataFrame(rows, index=pd.DatetimeIndex(index, name="ts"))
    return frame[frame["close"] > 0].sort_index()


__all__ = [
    "CALL",
    "CHAIN_STEP",
    "PUT",
    "SESSION_MINUTES_PER_DAY",
    "STRIKE_STEP",
    "LegQuote",
    "OptionChain",
    "SYNTHETIC_IV_FLOOR",
    "annualised_vol",
    "chain_iv",
    "leg_bars",
    "next_weekly_expiry",
    "seed_of",
    "smile_iv",
    "synthetic_chain",
    "trading_minutes_to",
]
