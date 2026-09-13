"""Option pricing for short-dated index options.

Black-Scholes, with two deliberate simplifications that are stated rather than
hidden:

* **The risk-free rate is zero.** On a same-day or next-day NIFTY option the
  carry term is worth a small fraction of a basis point of premium, and an
  invented rate would be a guess dressed up as a parameter.
* **Time is measured in trading minutes, not calendar time.** An option does not
  decay at the weekend, and a chain priced off the wall clock would show Friday's
  theta spreading over a week off-hours. ``TRADING_MINUTES_PER_YEAR`` is what
  makes a same-day option decay to nothing by 15:30 rather than by midnight.

What this module is *for*: the platform's whole argument is that a signal is only
worth acting on when the move it forecasts clears the cost of capturing it. A
long option has an unusually explicit version of that cost — you pay the premium,
and the underlying has to travel far enough for the delta to earn it back, before
theta eats it. :func:`breakeven_move_bps` is that number, and it is the reason
most single-leg long-premium scalps are not trades at all.
"""

from __future__ import annotations

from math import erf, exp, log, pi, sqrt

CALL = "CE"
PUT = "PE"

# NSE index session: 375 minutes a day, roughly 252 trading days a year.
TRADING_MINUTES_PER_DAY = 375
TRADING_DAYS_PER_YEAR = 252
TRADING_MINUTES_PER_YEAR = TRADING_MINUTES_PER_DAY * TRADING_DAYS_PER_YEAR

# Any shorter than this and the option is intrinsic; the floor only exists to
# keep a division by time from exploding on the last tick of the session.
MIN_YEARS = 1e-7


def years_from_minutes(minutes: float) -> float:
    """Trading minutes to years, which is the unit theta is quoted against here."""
    return max(float(minutes), 0.0) / TRADING_MINUTES_PER_YEAR


def _norm_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _norm_pdf(value: float) -> float:
    return exp(-0.5 * value * value) / sqrt(2.0 * pi)


def _d1_d2(spot: float, strike: float, iv: float, years: float) -> tuple[float, float]:
    span = max(iv * sqrt(max(years, MIN_YEARS)), 1e-9)
    d1 = (log(spot / strike) + 0.5 * span * span) / span
    return d1, d1 - span


def is_call(kind: str) -> bool:
    return str(kind).upper() in {"CE", "C", "CALL"}


def bs_price(spot: float, strike: float, iv: float, years: float, kind: str) -> float:
    """Premium per unit of the underlying."""
    if spot <= 0 or strike <= 0:
        return 0.0
    if years <= MIN_YEARS or iv <= 0:
        intrinsic = spot - strike if is_call(kind) else strike - spot
        return max(intrinsic, 0.0)

    d1, d2 = _d1_d2(spot, strike, iv, years)
    if is_call(kind):
        return spot * _norm_cdf(d1) - strike * _norm_cdf(d2)
    return strike * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def bs_delta(spot: float, strike: float, iv: float, years: float, kind: str) -> float:
    """Sensitivity of premium to a one-point move in the underlying.

    Signed, so a put's delta is negative and a short leg's is the negative of the
    long's — which is the sign a position's risk is built from.
    """
    if spot <= 0 or strike <= 0:
        return 0.0
    if years <= MIN_YEARS or iv <= 0:
        if is_call(kind):
            return 1.0 if spot > strike else 0.0
        return -1.0 if spot < strike else 0.0

    d1, _ = _d1_d2(spot, strike, iv, years)
    return _norm_cdf(d1) if is_call(kind) else _norm_cdf(d1) - 1.0


def bs_gamma(spot: float, strike: float, iv: float, years: float) -> float:
    """Change in delta per point of underlying. Same for a call and a put."""
    if spot <= 0 or strike <= 0 or years <= MIN_YEARS or iv <= 0:
        return 0.0
    d1, _ = _d1_d2(spot, strike, iv, years)
    span = max(iv * sqrt(max(years, MIN_YEARS)), 1e-9)
    return _norm_pdf(d1) / (spot * span)


def bs_theta_per_minute(spot: float, strike: float, iv: float, years: float, kind: str = CALL) -> float:
    """Premium lost per trading minute, negative for a long option.

    Identical for a call and a put with rates at zero, and no direction argument
    changes that — the sign a *position* carries comes from whether it is long or
    short, not from which side of the chain it sits on.

    This is the number that makes the recommendation honest: it is what a position
    pays for the privilege of waiting, and on a scalping horizon it is comparable
    to the move being waited for.
    """
    if spot <= 0 or strike <= 0 or years <= MIN_YEARS or iv <= 0:
        return 0.0
    d1, _ = _d1_d2(spot, strike, iv, years)
    per_year = -(spot * _norm_pdf(d1) * iv) / (2.0 * sqrt(max(years, MIN_YEARS)))
    return per_year / TRADING_MINUTES_PER_YEAR


def bs_vega_per_vol_point(spot: float, strike: float, iv: float, years: float) -> float:
    """Premium change per one percentage point of implied vol."""
    if spot <= 0 or strike <= 0 or years <= MIN_YEARS or iv <= 0:
        return 0.0
    d1, _ = _d1_d2(spot, strike, iv, years)
    return spot * _norm_pdf(d1) * sqrt(max(years, MIN_YEARS)) / 100.0


def atm_premium(spot: float, iv: float, years: float) -> float:
    """Premium of an at-the-money option, which is where the chain's money is."""
    return bs_price(spot, spot, iv, years, CALL)


def breakeven_move_bps(premium: float, delta: float, spot: float) -> float:
    """Size of the underlying move, in basis points, that a leg needs to break even.

    The premium divided by the delta it buys, expressed against spot — and when
    ``premium`` is taken to include the round-trip cost and the theta paid while
    holding, this is the whole hurdle a long-premium trade has to clear.

    The sign of ``delta`` is ignored, and deliberately: what a put needs is the
    same *distance* travelled in the other direction, so a caller holding a put's
    negative delta must not get an infinite requirement for a finite move. A
    near-zero delta does make it explode, which is correct rather than a defect —
    a far out-of-the-money option has to be carried by a move so large that its
    own premium is the only reason it is cheap.
    """
    magnitude = abs(float(delta))
    if magnitude <= 1e-9 or spot <= 0 or premium <= 0:
        return float("inf")
    return (premium / (magnitude * spot)) * 10_000.0


__all__ = [
    "CALL",
    "MIN_YEARS",
    "PUT",
    "TRADING_MINUTES_PER_DAY",
    "TRADING_MINUTES_PER_YEAR",
    "atm_premium",
    "breakeven_move_bps",
    "bs_delta",
    "bs_gamma",
    "bs_price",
    "bs_theta_per_minute",
    "bs_vega_per_vol_point",
    "is_call",
    "years_from_minutes",
]
