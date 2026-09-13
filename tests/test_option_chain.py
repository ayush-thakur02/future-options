"""Tests for option pricing, the chain, and the legs derived from it.

Option arithmetic is easy to get subtly wrong and hard to notice: a premium that
is 5% too cheap still looks like a premium, and a breakeven that should be 100bp
still reads as a number. These tests pin the identities (put-call parity, delta
bounds, intrinsic at expiry), the monotonicity that makes the panel readable, and
the one property the whole design rests on — that the premium decays across a
session because time passes, not because a drift term was chosen.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from core.calendar import IST
from core.settings import Settings
from plugins.sources.option_chain import pricing as px
from plugins.sources.option_chain.chain import (
    CALL,
    PUT,
    SYNTHETIC_IV_FLOOR,
    annualised_vol,
    chain_iv,
    leg_bars,
    next_weekly_expiry,
    smile_iv,
    synthetic_chain,
    trading_minutes_to,
)
from plugins.sources.option_chain.plugin import OptionChainSource
from plugins.sources.simulated.series import generate_candles

SPOT = 24_000.0
IV = 0.12
ONE_DAY = px.years_from_minutes(375)
NOW = datetime(2026, 3, 10, 10, 0, tzinfo=IST)


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    return generate_candles(days=3, seed=13)


@pytest.fixture(scope="module")
def source() -> OptionChainSource:
    return OptionChainSource(Settings())


# --------------------------------------------------------------------- pricing


@pytest.mark.parametrize("strike", [22_000.0, 23_500.0, 24_000.0, 24_500.0, 26_000.0])
def test_put_call_parity_holds_at_zero_rates(strike: float) -> None:
    """CE - PE = spot - strike when the rate is zero. The core identity of the model."""
    call = px.bs_price(SPOT, strike, IV, ONE_DAY, CALL)
    put = px.bs_price(SPOT, strike, IV, ONE_DAY, PUT)
    assert call - put == pytest.approx(SPOT - strike, rel=1e-6, abs=1e-6)


def test_atm_call_and_put_are_worth_the_same() -> None:
    call = px.bs_price(SPOT, SPOT, IV, ONE_DAY, CALL)
    put = px.bs_price(SPOT, SPOT, IV, ONE_DAY, PUT)
    assert call == pytest.approx(put, rel=1e-9)
    assert call == pytest.approx(px.atm_premium(SPOT, IV, ONE_DAY))


def test_premium_falls_with_distance_from_the_money() -> None:
    """A call is worth less the further out of the money it sits."""
    prices = [px.bs_price(SPOT, strike, IV, ONE_DAY, CALL) for strike in (23_000, 24_000, 25_000)]
    assert prices[0] > prices[1] > prices[2]


def test_premium_rises_with_implied_vol() -> None:
    low = px.bs_price(SPOT, SPOT, 0.08, ONE_DAY, CALL)
    high = px.bs_price(SPOT, SPOT, 0.30, ONE_DAY, CALL)
    assert high > low


def test_premium_is_intrinsic_at_expiry() -> None:
    itm = px.bs_price(24_500.0, 24_000.0, IV, 0.0, CALL)
    otm = px.bs_price(23_500.0, 24_000.0, IV, 0.0, CALL)
    put_itm = px.bs_price(23_500.0, 24_000.0, IV, 0.0, PUT)
    assert itm == pytest.approx(500.0)
    assert otm == 0.0
    assert put_itm == pytest.approx(500.0)


def test_deltas_are_bounded_and_meet_at_the_money() -> None:
    call_deltas = [px.bs_delta(SPOT, strike, IV, ONE_DAY, CALL) for strike in (22_000, 24_000, 26_000)]
    put_deltas = [px.bs_delta(SPOT, strike, IV, ONE_DAY, PUT) for strike in (22_000, 24_000, 26_000)]

    assert all(0.0 <= value <= 1.0 for value in call_deltas)
    assert all(-1.0 <= value <= 0.0 for value in put_deltas)
    assert px.bs_delta(SPOT, SPOT, IV, ONE_DAY, CALL) == pytest.approx(0.5, abs=0.01)
    assert px.bs_delta(SPOT, SPOT, IV, ONE_DAY, PUT) == pytest.approx(-0.5, abs=0.01)
    # Deep in the money a call moves one for one.
    assert px.bs_delta(SPOT, 20_000.0, IV, ONE_DAY, CALL) > 0.95


def test_gamma_is_positive_and_peaks_at_the_money() -> None:
    wings = px.bs_gamma(SPOT, 23_000.0, IV, ONE_DAY)
    middle = px.bs_gamma(SPOT, SPOT, IV, ONE_DAY)
    assert middle > 0 and wings > 0
    assert middle > wings


def test_theta_is_negative_and_worse_near_expiry() -> None:
    """The shorter the time left, the more premium burns per minute."""
    far = px.bs_theta_per_minute(SPOT, SPOT, IV, px.years_from_minutes(5 * 375))
    near = px.bs_theta_per_minute(SPOT, SPOT, IV, px.years_from_minutes(375))
    last_hour = px.bs_theta_per_minute(SPOT, SPOT, IV, px.years_from_minutes(60))
    assert far < 0 and near < 0 and last_hour < 0
    assert abs(last_hour) > abs(near) > abs(far)


def test_vega_is_positive_and_peaks_at_the_money() -> None:
    assert px.bs_vega_per_vol_point(SPOT, SPOT, IV, ONE_DAY) > 0
    assert px.bs_vega_per_vol_point(SPOT, SPOT, IV, ONE_DAY) > px.bs_vega_per_vol_point(
        SPOT, 27_000.0, IV, ONE_DAY
    )


def test_years_from_minutes_counts_trading_time() -> None:
    assert px.years_from_minutes(375) == pytest.approx(1 / 252)
    assert px.years_from_minutes(0) == 0.0
    assert px.years_from_minutes(-5) == 0.0


# ---------------------------------------------------------------- breakeven


def test_breakeven_move_scales_with_premium_and_delta() -> None:
    cheap = px.breakeven_move_bps(50.0, 0.5, SPOT)
    dear = px.breakeven_move_bps(100.0, 0.5, SPOT)
    assert dear == pytest.approx(cheap * 2)

    dull = px.breakeven_move_bps(100.0, 0.25, SPOT)
    assert dull == pytest.approx(px.breakeven_move_bps(100.0, 0.5, SPOT) * 2)


def test_breakeven_is_the_same_distance_for_a_put_as_for_a_call() -> None:
    """A put's delta is negative but the travel it needs is the same size.

    Returning infinity for a put — which is what taking the sign at face value
    did — would make every downside leg look untradeable.
    """
    assert px.breakeven_move_bps(100.0, -0.5, SPOT) == px.breakeven_move_bps(100.0, 0.5, SPOT)


def test_breakeven_is_infinite_without_a_delta() -> None:
    assert px.breakeven_move_bps(100.0, 0.0, SPOT) == float("inf")
    assert px.breakeven_move_bps(100.0, 1e-12, SPOT) == float("inf")
    assert px.breakeven_move_bps(0.0, 0.5, SPOT) == float("inf")


def test_atm_breakeven_is_the_number_the_platform_argues_about(source) -> None:
    """A one-day ATM long needs a move two orders of magnitude beyond a 1m median.

    This is the single most important number the options board reports, and the
    reason its default recommendation is to do nothing.
    """
    quote = synthetic_chain(SPOT, NOW, atm_iv=IV).atm(CALL)
    required = quote.breakeven_move_bps()
    assert required > 40, f"expected a demanding breakeven, got {required:.1f}bp"
    assert required < 200


# -------------------------------------------------------------------- clock


def test_trading_minutes_counts_a_full_session() -> None:
    start = datetime(2026, 3, 10, 9, 15, tzinfo=IST)
    end = datetime(2026, 3, 10, 15, 30, tzinfo=IST)
    assert trading_minutes_to(start, end) == pytest.approx(375.0)


def test_trading_minutes_ignores_closed_hours() -> None:
    """Nothing decays overnight, so nothing is counted overnight."""
    close = datetime(2026, 3, 10, 15, 30, tzinfo=IST)
    open_next = datetime(2026, 3, 11, 9, 15, tzinfo=IST)
    assert trading_minutes_to(close, open_next) == 0.0


def test_trading_minutes_spans_a_weekend_correctly() -> None:
    friday_close = datetime(2026, 3, 13, 15, 30, tzinfo=IST)
    monday_open = datetime(2026, 3, 16, 9, 15, tzinfo=IST)
    assert trading_minutes_to(friday_close, monday_open) == 0.0
    assert trading_minutes_to(friday_close, datetime(2026, 3, 16, 10, 15, tzinfo=IST)) == pytest.approx(60.0)


def test_time_to_expiry_never_goes_negative() -> None:
    past = datetime(2026, 3, 1, 10, 0, tzinfo=IST)
    assert trading_minutes_to(NOW, past) == 0.0


def test_next_weekly_expiry_lands_on_the_configured_weekday() -> None:
    expiry = next_weekly_expiry(NOW, weekday=1)
    assert expiry.weekday() == 1
    assert expiry > NOW
    assert expiry.hour == 15 and expiry.minute == 30


def test_next_weekly_expiry_rolls_past_one_that_has_closed() -> None:
    tuesday_close = datetime(2026, 3, 10, 15, 45, tzinfo=IST)
    assert next_weekly_expiry(tuesday_close, weekday=1) > tuesday_close
    assert (next_weekly_expiry(tuesday_close, weekday=1) - tuesday_close).days >= 6


# --------------------------------------------------------------- vol & smile


def test_annualised_vol_is_positive_and_bounded() -> None:
    closes = pd.Series(np.linspace(24_000, 24_100, 200))
    assert 0.02 <= annualised_vol(closes) <= 1.5


def test_annualised_vol_survives_a_flat_series() -> None:
    assert annualised_vol(pd.Series([24_000.0] * 50)) >= 0.02
    assert annualised_vol(pd.Series([24_000.0])) == 0.12


def test_chain_iv_has_a_floor_above_the_generated_series_realised_vol(bars) -> None:
    """The bundled series is quieter than any real NIFTY session."""
    realised = annualised_vol(bars["close"])
    assert chain_iv(bars["close"]) == pytest.approx(SYNTHETIC_IV_FLOOR)
    assert chain_iv(bars["close"]) >= realised


def test_smile_puts_a_premium_on_the_wings() -> None:
    at_money = smile_iv(0.12, 0.0)
    wing = smile_iv(0.12, 400.0)
    far_wing = smile_iv(0.12, 800.0)
    assert at_money == pytest.approx(0.12)
    assert wing > at_money
    assert far_wing > wing


# -------------------------------------------------------------------- chain


def test_chain_is_centred_on_the_money(source) -> None:
    chain = synthetic_chain(24_137.0, NOW, atm_iv=IV, width=5, strike_step=50)
    assert chain.atm_strike == 24_150.0
    strikes = chain.strikes()
    assert len(strikes) == 11
    assert strikes[0] == 24_150.0 - 5 * 50
    assert strikes[-1] == 24_150.0 + 5 * 50


def test_chain_quotes_both_sides_of_every_strike(source) -> None:
    chain = synthetic_chain(SPOT, NOW, atm_iv=IV, width=4)
    for strike in chain.strikes():
        assert chain.at(strike, CALL) is not None
        assert chain.at(strike, PUT) is not None


def test_chain_open_interest_leans_with_the_market() -> None:
    """Calls build above spot and puts below it, which is what makes PCR mean anything."""
    chain = synthetic_chain(SPOT, NOW, atm_iv=IV, width=8)
    above = [quote.oi for quote in chain.side(CALL) if quote.strike > SPOT]
    below = [quote.oi for quote in chain.side(CALL) if quote.strike < SPOT]
    puts_below = [quote.oi for quote in chain.side(PUT) if quote.strike < SPOT]
    puts_above = [quote.oi for quote in chain.side(PUT) if quote.strike > SPOT]
    assert np.mean(above) > np.mean(below)
    assert np.mean(puts_below) > np.mean(puts_above)


def test_chain_totals_report_a_plausible_pcr() -> None:
    totals = synthetic_chain(SPOT, NOW, atm_iv=IV).totals()
    assert 0.3 < totals["pcr"] < 3.0
    assert totals["call_oi"] > 0 and totals["put_oi"] > 0


def test_chain_reports_greeks_with_the_right_signs() -> None:
    chain = synthetic_chain(SPOT, NOW, atm_iv=IV)
    call, put = chain.atm(CALL), chain.atm(PUT)
    assert call.delta > 0 and put.delta < 0
    assert call.gamma > 0 and put.gamma > 0
    assert call.theta_per_minute < 0 and put.theta_per_minute < 0
    assert call.money_per_minute > 0


def test_chain_iv_rises_away_from_the_money() -> None:
    chain = synthetic_chain(SPOT, NOW, atm_iv=0.12, width=8)
    at_money = chain.atm(CALL).iv
    wing = chain.at(SPOT + 8 * 50, CALL).iv
    assert wing > at_money


def test_moneyness_is_signed_and_in_basis_points() -> None:
    chain = synthetic_chain(SPOT, NOW, atm_iv=IV, width=4)
    above = chain.at(SPOT + 200, CALL)
    below = chain.at(SPOT - 200, PUT) if chain.at(SPOT - 200, PUT) else chain.at(SPOT - 200, CALL)
    assert above.moneyness > 0
    assert below.moneyness < 0
    assert above.moneyness == pytest.approx(200 / SPOT * 10_000, rel=1e-6)


# ----------------------------------------------------------------- leg bars


def test_leg_bars_look_like_candles(bars, source) -> None:
    leg = leg_bars(bars.tail(60), 24_150.0, CALL, source.expiry())
    assert not leg.empty
    assert list(leg.columns)[:6] == ["open", "high", "low", "close", "volume", "oi"]
    assert (leg["close"] >= 0).all()
    assert (leg["high"] >= leg[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (leg["low"] <= leg[["open", "close"]].min(axis=1) + 1e-9).all()


def test_leg_bars_carry_greeks_and_volume(bars, source) -> None:
    leg = leg_bars(bars.tail(60), 24_150.0, CALL, source.expiry())
    for column in ("delta", "gamma", "theta_per_minute", "iv", "minutes_to_expiry", "spot"):
        assert column in leg.columns
        assert np.isfinite(leg[column].to_numpy(dtype="float64")).all()
    # The index has no volume to inherit, so a leg that carries none would switch
    # off every activity feature built on top of it.
    assert (leg["volume"] > 0).all()
    assert (leg["oi"] > 0).all()


def test_leg_premium_decays_when_the_index_does_not_move() -> None:
    """The strongest test of the clock: flat spot, falling premium, because time passes."""
    stamps = pd.date_range("2026-03-10 09:15", periods=60, freq="1min", tz=IST)
    flat = pd.DataFrame(
        {"open": 24_000.0, "high": 24_000.5, "low": 23_999.5, "close": 24_000.0},
        index=stamps,
    )
    expiry = datetime(2026, 3, 10, 15, 30, tzinfo=IST)
    leg = leg_bars(flat, 24_000.0, CALL, expiry, atm_iv=0.12)

    assert leg["close"].iloc[-1] < leg["close"].iloc[0]
    assert (leg["theta_per_minute"] < 0).all()
    # And the decay accelerates as expiry approaches, as it does in reality.
    early = leg["close"].iloc[0] - leg["close"].iloc[29]
    late = leg["close"].iloc[30] - leg["close"].iloc[59]
    assert late > early


def test_leg_premium_tracks_the_index_direction(bars, source) -> None:
    """A call's premium must rise when the index does, bar for bar."""
    window = bars.tail(40)
    leg = leg_bars(window, 24_150.0, CALL, source.expiry())
    common = leg.index.intersection(window.index)
    index_move = window.loc[common, "close"].diff().dropna()
    premium_move = leg.loc[common, "close"].diff().dropna()
    agree = (np.sign(index_move) == np.sign(premium_move)).mean()
    assert agree > 0.9, f"premium and index disagreed on {1 - agree:.0%} of bars"


def test_leg_bars_are_reproducible(bars, source) -> None:
    """Two runs over the same bars must produce the same chain, or a bug is not findable twice."""
    first = leg_bars(bars.tail(30), 24_150.0, CALL, source.expiry())
    second = leg_bars(bars.tail(30), 24_150.0, CALL, source.expiry())
    pd.testing.assert_frame_equal(first, second)


def test_leg_bars_handle_an_empty_series(source) -> None:
    assert leg_bars(generate_candles(days=1, seed=1).iloc[0:0], 24_000.0, CALL, source.expiry()).empty


# ------------------------------------------------------------------- source


def test_source_builds_both_legs_at_one_strike(source, bars) -> None:
    legs = source.legs(bars.tail(120))
    assert set(legs) == {CALL, PUT}
    assert legs[CALL].strike == legs[PUT].strike
    assert legs[CALL].label == "CALL" and legs[PUT].label == "PUT"
    assert legs[CALL].quote.delta > 0 > legs[PUT].quote.delta


def test_source_reports_the_greeks_the_panel_needs(source, bars) -> None:
    greeks = source.legs(bars.tail(120))[CALL].greeks
    for key in ("premium", "delta", "gamma", "theta_per_minute", "vega", "iv", "oi", "breakeven_bps"):
        assert key in greeks
        assert np.isfinite(greeks[key])


def test_source_quote_matches_the_chain_it_comes_from(source) -> None:
    chain = source.chain(SPOT, NOW, bars=generate_candles(days=1, seed=2))
    quote = source.quote(CALL, SPOT, SPOT, NOW, atm_iv=chain.atm(CALL).iv)
    assert quote.premium == pytest.approx(chain.atm(CALL).premium, rel=1e-9)


def test_source_expiry_is_cached_until_it_passes(source) -> None:
    first = source.expiry(NOW)
    assert source.expiry(NOW + timedelta(minutes=5)) == first
    assert source.expiry(first + timedelta(minutes=1)) > first


def test_source_minutes_to_expiry_counts_down(source) -> None:
    later = source.minutes_to_expiry(NOW)
    sooner = source.minutes_to_expiry(NOW + timedelta(hours=1))
    assert sooner < later


def test_source_returns_nothing_without_bars(source) -> None:
    assert source.legs(pd.DataFrame()) == {}


def test_leg_breakeven_accepts_the_cost_of_trading_it(source, bars) -> None:
    """Brokerage and slippage are not a rounding error on a one-minute horizon."""
    leg = source.legs(bars.tail(60))[CALL]
    bare = leg.quote.breakeven_move_bps()
    with_cost = leg.quote.breakeven_move_bps(extra_cost=2.0)
    assert with_cost > bare
    assert with_cost - bare == pytest.approx((2.0 / (abs(leg.quote.delta) * leg.quote.spot)) * 10_000, rel=1e-6)
