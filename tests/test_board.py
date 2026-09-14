"""Tests for the call/index/put board and the verdict arithmetic.

The board is where three instruments, one clock and one cost argument meet, so
the failures worth guarding against are the ones that look plausible: a leg
drawn on a scale that flatters it, a verdict computed against the wrong price,
a shared projection that quietly ties two instruments together.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from rich.console import Console
from rich.text import Text

from core.calendar import IST
from core.settings import Settings
from core.types import Tick
from kernel import Kernel
from plugins.advisory.breakeven_gate.gate import (
    FLAT,
    LONG,
    SHORT,
    assess_option,
    headline,
    verdict_for_option,
    verdict_for_underlying,
)
from plugins.advisory.breakeven_gate.plugin import BreakevenAdvisor
from plugins.renderers.terminal import TerminalRenderer
from plugins.sources.simulated.series import generate_candles
from runtime.board import MarketBoard

SPOT = 24_000.0


@pytest.fixture(scope="module")
def bars():
    return generate_candles(days=3, seed=29)


@pytest.fixture(scope="module")
def board(bars):
    built = MarketBoard(Kernel.bootstrap(Settings()), bar_minutes=1).build(bars)
    return built


# --------------------------------------------------------------- gate: long


def test_requirement_is_premium_over_delta_in_basis_points() -> None:
    """The central identity: what the underlying must do to pay for the premium."""
    assessment = assess_option(
        premium=120.0,
        delta=0.5,
        spot=SPOT,
        theta_per_minute=0.0,
        horizon_minutes=0.0,
        cost_rate=0.0,
    )
    assert assessment.required_move_bps == pytest.approx(100.0, rel=1e-6)


def test_costs_and_theta_both_raise_the_requirement() -> None:
    bare = assess_option(120.0, 0.5, SPOT, 0.0, 0.0, cost_rate=0.0)
    with_cost = assess_option(120.0, 0.5, SPOT, 0.0, 0.0, cost_rate=0.02)
    with_theta = assess_option(120.0, 0.5, SPOT, 0.5, 3.0, cost_rate=0.0)

    assert with_cost.required_move_bps > bare.required_move_bps
    assert with_theta.required_move_bps > bare.required_move_bps
    assert with_cost.cost_bps > 0
    assert with_theta.theta_cost_bps > 0


def test_theta_billed_over_a_horizon_cannot_exceed_the_premium() -> None:
    """A leg cannot lose more than it is worth, however long it is held.

    Unclamped, holding a 10-rupee option for a thousand minutes at 5 rupees a
    minute would require a move no market makes — and it would be arithmetic
    rather than a finding.
    """
    premium = 10.0
    assessment = assess_option(premium, 0.5, SPOT, theta_per_minute=5.0, horizon_minutes=1_000.0)
    theta_charged = assessment.premium_cost - premium * (1 + 0.012)
    assert theta_charged <= premium + 1e-9


def test_a_higher_delta_lowers_the_requirement() -> None:
    dull = assess_option(120.0, 0.2, SPOT, 0.0, 0.0)
    sharp = assess_option(120.0, 0.6, SPOT, 0.0, 0.0)
    assert sharp.required_move_bps < dull.required_move_bps


def test_a_legacy_zero_premium_is_not_tradeable() -> None:
    assessment = assess_option(0.0, 0.5, SPOT, 0.0, 0.0)
    assert assessment.required_move_bps == float("inf")
    assert not assessment.is_reachable


def test_an_unreachable_requirement_is_flagged() -> None:
    assert not assess_option(10_000.0, 0.01, SPOT, 0.0, 0.0).is_reachable
    assert assess_option(120.0, 0.5, SPOT, 0.0, 0.0).is_reachable


# ------------------------------------------------------------- gate: verdict


def assessment(required: float = 100.0) -> object:
    return assess_option(required / 10_000.0 * 0.5 * SPOT, 0.5, SPOT, 0.0, 0.0, cost_rate=0.0)


def test_a_leg_on_the_right_side_that_clears_its_cost_is_a_long() -> None:
    verdict = verdict_for_option(
        "CALL", "CE", assessment(100.0), projected_move_bps=250.0,
        iv=0.12, realised_vol=0.12, strike=SPOT, spot=SPOT, strike_step=50.0,
    )
    assert verdict.action == LONG
    assert verdict.edge_bps == pytest.approx(150.0, rel=1e-3)
    assert "needs 100bp" in verdict.reason


def test_a_leg_on_the_right_side_that_cannot_pay_is_flat() -> None:
    verdict = verdict_for_option(
        "CALL", "CE", assessment(100.0), projected_move_bps=8.0,
        iv=0.12, realised_vol=0.12, strike=SPOT, spot=SPOT, strike_step=50.0,
    )
    assert verdict.action == FLAT
    assert verdict.edge_bps < 0


def test_a_leg_on_the_wrong_side_is_flat_without_rich_vol() -> None:
    verdict = verdict_for_option(
        "PUT", "PE", assessment(100.0), projected_move_bps=250.0,
        iv=0.12, realised_vol=0.12, strike=SPOT, spot=SPOT, strike_step=50.0,
    )
    assert verdict.action == FLAT
    assert "other way" in verdict.reason


def test_rich_vol_and_the_wrong_side_makes_a_short() -> None:
    verdict = verdict_for_option(
        "PUT", "PE", assessment(100.0), projected_move_bps=250.0,
        iv=0.30, realised_vol=0.15, strike=SPOT - 200.0, spot=SPOT, strike_step=50.0,
    )
    assert verdict.action == SHORT
    assert "IV" in verdict.reason


def test_a_short_needs_the_strike_out_of_the_money() -> None:
    """At the money there is no premium cushion, so the gate refuses to write it."""
    verdict = verdict_for_option(
        "PUT", "PE", assessment(100.0), projected_move_bps=250.0,
        iv=0.30, realised_vol=0.15, strike=SPOT, spot=SPOT, strike_step=50.0,
    )
    assert verdict.action == FLAT


def test_underlying_verdict_uses_the_round_trip_hurdle() -> None:
    trade = verdict_for_underlying("INDEX", projected_move_bps=20.0, hurdle_bps=3.9, conviction=0.4)
    assert trade.action == LONG
    assert trade.required_move_bps == pytest.approx(3.9)

    idle = verdict_for_underlying("INDEX", projected_move_bps=20.0, hurdle_bps=3.9, conviction=0.0)
    assert idle.action == FLAT
    assert idle.reason == "no directional view"

    short = verdict_for_underlying("INDEX", projected_move_bps=1.0, hurdle_bps=3.9, conviction=0.4)
    assert short.action == FLAT


def test_headline_prefers_a_long_over_a_richer_short() -> None:
    """A short's edge is not a negative long's: ranking them together picks the short."""
    long_leg = verdict_for_option(
        "CALL", "CE", assessment(100.0), projected_move_bps=250.0,
        iv=0.12, realised_vol=0.12, strike=SPOT, spot=SPOT, strike_step=50.0,
    )
    short_leg = verdict_for_option(
        "PUT", "PE", assessment(100.0), projected_move_bps=250.0,
        iv=0.40, realised_vol=0.15, strike=SPOT - 300.0, spot=SPOT, strike_step=50.0,
    )
    assert headline([short_leg, long_leg]).startswith("CALL LONG")


def test_headline_reports_no_trade_when_nothing_qualifies() -> None:
    flat = verdict_for_option(
        "CALL", "CE", assessment(100.0), projected_move_bps=1.0,
        iv=0.12, realised_vol=0.12, strike=SPOT, spot=SPOT, strike_step=50.0,
    )
    assert "no trade" in headline([flat])


# -------------------------------------------------------------------- board


def test_board_has_the_three_legs_at_one_strike(board) -> None:
    assert [leg.label for leg in board.legs] == ["INDEX", "CALL", "PUT"]
    assert board.leg("CALL").strike == board.leg("PUT").strike
    assert board.leg("CALL").kind == "CE"
    assert board.leg("PUT").kind == "PE"


def test_every_instrument_gets_its_own_projection(bars) -> None:
    """A shared projection instance would tie three instruments' paths together.

    The kernel memoises by handle, so this is the failure the board's use of
    ``new`` exists to prevent, and it would look like three charts that move in
    perfect lockstep.
    """
    board = MarketBoard(Kernel.bootstrap(Settings()), bar_minutes=1).build(bars)
    forecasters = [leg.engine.forecaster for leg in board.legs]
    assert all(forecaster is not None for forecaster in forecasters)
    assert len({id(forecaster) for forecaster in forecasters}) == len(board.legs)

    trackers = [id(forecaster.tracker) for forecaster in forecasters]
    assert len(set(trackers)) == len(board.legs)


def test_every_leg_starts_with_a_path_to_draw(board) -> None:
    for leg in board.legs:
        assert leg.engine.projections, f"{leg.label} opened with nothing to draw"
        assert len(leg.engine.projections) == 3


def test_a_leg_tick_follows_the_index_tick(board) -> None:
    """The leg has no feed of its own; its premium is a function of the index."""
    call = board.leg("CALL")
    put = board.leg("PUT")
    before_call, before_put = call.engine.last_price, put.engine.last_price
    spot = board.index_engine.last_price

    board.on_tick(Tick(ts=datetime.now(IST), ltp=spot * 1.004))

    assert call.engine.last_price > before_call, "a call premium must rise with the index"
    assert put.engine.last_price < before_put, "a put premium must fall with the index"


def test_the_index_engine_is_the_board_s_first_leg(board) -> None:
    assert board.index_engine is board.leg("INDEX").engine


def test_board_snapshot_carries_verdicts_and_a_headline(board) -> None:
    snapshot = board.snapshot()
    assert snapshot.spot > 0
    assert len(snapshot.legs) == 3
    assert snapshot.headline

    for leg in snapshot.legs:
        assert leg.verdict is not None, f"{leg.label} has no verdict"
        assert leg.verdict.required_move_bps >= 0
        assert leg.verdict.action in {"LONG", "SHORT", "FLAT"}


def test_option_verdicts_are_measured_against_the_underlying(board) -> None:
    """A requirement in the tens of thousands of basis points means the premium
    was divided by its own delta against itself — the bug this pins shut.
    """
    snapshot = board.snapshot()
    for leg in snapshot.option_legs():
        assert leg.verdict.required_move_bps < 1_000.0, (
            f"{leg.label} needs {leg.verdict.required_move_bps:,.0f}bp, "
            f"which is not a move an index makes"
        )


def test_leg_greeks_are_reported_for_the_panel(board) -> None:
    for leg in board.snapshot().option_legs():
        for key in ("premium", "delta", "theta_per_minute", "iv", "strike"):
            assert key in leg.greeks
    assert board.snapshot().leg("CALL").greeks["delta"] > 0


def test_chain_summary_reaches_the_board(board) -> None:
    chain = board.snapshot().chain
    assert chain["call_oi"] > 0 and chain["put_oi"] > 0
    assert "expiry" in chain


def test_legs_roll_when_the_spot_walks_away(bars) -> None:
    board = MarketBoard(Kernel.bootstrap(Settings()), bar_minutes=1).build(bars)
    call = board.leg("CALL")
    original = call.strike

    assert board.roll_if_needed(original + call.step) == [], "one strike away is not a roll"
    rolled = board.roll_if_needed(original + call.step * 4)

    assert rolled, "four strikes away must roll the leg"
    assert call.strike != original
    assert call.rolls == 1


def test_a_roll_resets_the_scorecard(bars) -> None:
    """A projection about one strike must never be scored against another's bars."""
    board = MarketBoard(Kernel.bootstrap(Settings()), bar_minutes=1).build(bars)
    call = board.leg("CALL")
    call.engine.forecaster.tracker.record(call.engine.projections, call.engine.last_price)
    assert call.engine.forecaster.tracker.pending > 0

    board.roll_if_needed(call.strike + call.step * 4)
    assert call.engine.forecaster.tracker.pending == 0
    assert call.engine.projections == []


def test_board_without_a_chain_still_shows_the_index() -> None:
    kernel = Kernel(Settings())
    for entry in Kernel.bootstrap(Settings()).registry.find():
        if entry.handle == "source:option_chain":
            continue
        kernel.register(entry)

    board = MarketBoard(kernel, bar_minutes=1)
    board.build(generate_candles(days=1, seed=4))
    assert [leg.label for leg in board.legs] == ["INDEX"]
    assert board.snapshot().headline


def test_board_refresh_and_publish_satisfy_the_clock(board) -> None:
    """The nowcast loop drives a board exactly as it drives an engine."""
    assert board.refresh_projection() is True
    board.publish()
    assert board.describe()


def test_advisor_attaches_verdicts_to_a_board(board) -> None:
    advisor = BreakevenAdvisor(horizon_bars=3, hurdle_bps=4.0)
    snapshot = advisor.advise(board.snapshot(), bar_minutes=1)
    assert snapshot.headline
    for leg in snapshot.legs:
        assert leg.verdict is not None
        assert leg.verdict.label == leg.label


def test_advisor_skips_a_board_leg_it_has_no_evidence_for() -> None:
    """A board of one index is a valid board; the advisor must not require legs."""
    from core.types import BoardSnapshot, LegSnapshot, MarketSnapshot

    engine_snapshot = MarketSnapshot(
        ts=datetime.now(IST),
        symbol="NIFTY 50",
        last_price=SPOT,
        prev_close=SPOT,
        candles=generate_candles(days=1, seed=2).tail(10),
        conviction=0.2,
    )
    board = BoardSnapshot(
        ts=datetime.now(IST),
        symbol="NIFTY 50",
        spot=SPOT,
        legs=[LegSnapshot(label="INDEX", kind="index", snapshot=engine_snapshot)],
    )
    advised = BreakevenAdvisor(hurdle_bps=4.0).advise(board, bar_minutes=1)
    assert advised.legs[0].verdict is not None
    assert "no trade" in advised.headline or "INDEX" in advised.headline


# ---------------------------------------------------------------- rendering


def render_board(board, width: int, height: int) -> list[str]:
    import io

    renderer = TerminalRenderer(symbol="NIFTY 50", timeframe="1m")
    renderer.console = Console(width=width, height=height)
    buffer = io.StringIO()
    Console(width=width, height=height, file=buffer, force_terminal=False).print(
        renderer.build_board(board.snapshot(), "live")
    )
    return buffer.getvalue().splitlines()


@pytest.mark.parametrize(("width", "height"), [(140, 44), (120, 40), (100, 36), (200, 60)])
def test_board_never_exceeds_the_console_width(board, width: int, height: int) -> None:
    lines = render_board(board, width, height)
    assert lines, "the board rendered nothing"
    for index, line in enumerate(lines):
        assert len(line) <= width, f"line {index} is {len(line)} chars, console is {width}"


@pytest.mark.parametrize(("width", "height"), [(140, 44), (120, 40), (200, 60)])
def test_board_draws_all_three_legs(board, width: int, height: int) -> None:
    rendered = "\n".join(render_board(board, width, height))
    assert "INDEX" in rendered
    assert "CALL" in rendered
    assert "PUT" in rendered
    assert "forward results" in rendered
    assert "plugin strategies" in rendered


def test_prediction_view_draws_paths_and_algorithm_matrix(board) -> None:
    import io

    frame = board.snapshot()
    for leg in frame.legs:
        leg.snapshot.research["ai"] = {
            "signals": {
                1: {
                    "algorithms": [
                        {"name": "ftrl_proximal", "p_up": 0.64, "trust_score": 0.42},
                        {"name": "adaptive_knn", "p_up": 0.57, "trust_score": 0.31},
                    ]
                }
            }
        }
    renderer = TerminalRenderer(symbol="NIFTY 50", timeframe="1m")
    renderer.view = "prediction"
    renderer.console = Console(width=140, height=44)
    buffer = io.StringIO()
    Console(width=140, height=44, file=buffer, force_terminal=False).print(
        renderer.build_board(frame, "live")
    )
    rendered = buffer.getvalue()
    assert "actual → projected trajectory" in rendered
    assert "algorithm agreement" in rendered
    assert "ftrl_proximal" in rendered
    assert "adaptive_knn" in rendered


def test_leg_panels_are_marked_blue_when_a_path_is_drawn(board) -> None:
    """The border says "this chart has a projection" at a glance, per leg."""
    from plugins.renderers.terminal import board as board_panels

    leg = board.snapshot().leg("CALL")
    panel = board_panels.leg_panel(leg, 44, 10, "1m")
    assert leg.snapshot.projections
    assert panel.border_style == "bright_blue"


def test_projected_leg_candles_are_blue_and_never_directional(board) -> None:
    """A projected premium must not be able to pass for a printed one."""
    from plugins.renderers.terminal.candles import render_chart

    leg = board.snapshot().leg("CALL")
    width = 40
    chart = render_chart(leg.snapshot.candles, leg.snapshot.projections, width=width, height=10)

    for line in chart.candle_lines:
        row = Text.from_markup(line)
        styles = [
            str(span.style)
            for span in row.spans
            if span.start < width and span.end > width - 3 and span.style is not None
        ]
        for style in styles:
            assert "green" not in style and "red" not in style


def test_a_narrow_board_drops_the_axis_rather_than_wrapping(board) -> None:
    """Three panels across a small terminal is where the axis has to give way."""
    lines = render_board(board, 90, 30)
    for line in lines:
        assert len(line) <= 90, f"a row overflowed: {line[:100]!r}"


def test_board_renders_with_a_single_leg(bars) -> None:
    kernel = Kernel(Settings())
    for entry in Kernel.bootstrap(Settings()).registry.find():
        if entry.handle == "source:option_chain":
            continue
        kernel.register(entry)
    single = MarketBoard(kernel, bar_minutes=1)
    single.build(bars.tail(120))

    rendered = "\n".join(render_board(single, 120, 36))
    assert "INDEX" in rendered
    assert "CALL" not in rendered


def test_board_strip_reports_the_numbers_that_decide_the_trade(board) -> None:
    from plugins.renderers.terminal.board import _strip

    text = _strip(board.snapshot().leg("CALL"))
    plain = Text.from_markup(text.markup).plain
    assert "Δ" in plain and "θ/min" in plain and "needs" in plain


def test_leg_strip_for_the_index_reports_conviction(board) -> None:
    from plugins.renderers.terminal.board import _strip

    plain = Text.from_markup(_strip(board.snapshot().leg("INDEX")).markup).plain
    assert "conv" in plain


def test_the_board_carries_only_the_history_the_engines_use(bars) -> None:
    """A cold start must not price or warm more bars than anything reads.

    The engines discard everything past their feature window and the charts show a
    fraction of it, so a board built from a multi-year cache would do seconds of
    work on three instruments and throw all of it away.
    """
    from runtime.engine import FEATURE_WINDOW

    long_history = generate_candles(days=60, seed=44)
    assert len(long_history) > FEATURE_WINDOW

    built = MarketBoard(Kernel.bootstrap(Settings()), bar_minutes=1).build(long_history)

    assert len(built.index_bars) == FEATURE_WINDOW
    for leg in built.legs:
        assert len(leg.engine.history) <= FEATURE_WINDOW


def test_opening_the_board_is_not_slow() -> None:
    """A guard against a return to a forty-second startup, with a wide margin.

    The loose bound is the point: it fails on seconds, not on milliseconds, so it
    catches a regression to the loop-per-bar behaviour without flaking.
    """
    import time

    from plugins.sources.simulated.series import generate_candles as candles

    history = candles(days=20, seed=45)
    started = time.perf_counter()
    board = MarketBoard(Kernel.bootstrap(Settings()), bar_minutes=1).build(history)
    elapsed = time.perf_counter() - started

    assert len(board.legs) == 3
    assert elapsed < 5.0, f"opening the board took {elapsed:.1f}s"
