"""Tests for the money simulator.

The failures worth guarding against here are the ones that produce a *plausible*
number rather than an error:

* rupees-per-point silently becoming rupees-per-notional, which turns a Rs 65
  move into a Rs 1.5 million one,
* a stop filling at whatever price the next refresh happened to see, which makes
  every stop look like a catastrophe,
* a put reported as disagreeing with every source that agrees on it,
* an option hurdle quoted without the decay it pays while being held,
* a reserve that tops a wallet up forever, or one that never gives the money back.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from backtest.costs import CostModel
from core.settings import Settings
from core.types import Direction, ForecastCandle, Signal
from plugins.advisory.money_simulator.ledger import SimulationLedger
from plugins.advisory.money_simulator.policy import (
    BUY,
    HOLD,
    SELL,
    DecisionPolicy,
    LegState,
    PolicyWeights,
)
from plugins.advisory.money_simulator.simulator import MoneySimulator, SimulatorConfig
from plugins.advisory.money_simulator.wallet import Reserve, Wallet

START = datetime(2026, 9, 15, 10, 0, 0)

# The user's own worked example: 65 units, so a point is worth exactly 65 rupees.
LOT = 65


def costs() -> CostModel:
    return CostModel(slippage_bps=0.5, lot_size=LOT)


def projection(anchor: float, bps: float, bars: int = 3) -> tuple[ForecastCandle, ...]:
    """A straight projected path covering ``bps`` in total over ``bars`` bars."""
    out = []
    price = anchor
    per_bar = bps / max(bars, 1)
    for index in range(bars):
        follow = price * (1 + per_bar / 10_000.0)
        out.append(
            ForecastCandle(
                ts=START + timedelta(minutes=index + 1),
                horizon=index + 1,
                open=price,
                high=max(price, follow),
                low=min(price, follow),
                close=follow,
            )
        )
        price = follow
    return tuple(out)


def signals(score: float) -> tuple[Signal, ...]:
    return (
        Signal(
            ts=START,
            strategy="ema_trend",
            direction=Direction.UP if score > 0 else Direction.DOWN,
            strength=abs(score),
            reason="test",
            meta={"state": "ACTIVE", "trust_score": 0.5, "raw_score": score},
        ),
    )


def index_state(price: float, bps: float = 8.0, atr: float = 20.0) -> LegState:
    return LegState(
        label="INDEX",
        kind="index",
        instrument="NSE_INDEX|Nifty 50",
        price=price,
        spot=price,
        index_beta=1.0,
        signals=signals(0.6 if bps >= 0 else -0.6),
        projections=projection(price, bps),
        index_projections=projection(price, bps),
        atr=atr,
        micro=0.2 if bps >= 0 else -0.2,
    )


def call_state(premium: float, spot: float, delta: float = 0.5, bps: float = 8.0) -> LegState:
    return LegState(
        label="CALL",
        kind="CE",
        instrument="SIM|NIFTY:24000:CE",
        price=premium,
        spot=spot,
        index_beta=1.0,
        signals=signals(0.6 if bps >= 0 else -0.6),
        projections=projection(spot, bps),
        index_projections=projection(spot, bps),
        atr=4.0,
        micro=0.2 if bps >= 0 else -0.2,
        greeks={"delta": delta, "premium": premium, "theta_per_minute": 0.0},
    )


def put_state(premium: float, spot: float, delta: float = -0.5, bps: float = -8.0) -> LegState:
    return LegState(
        label="PUT",
        kind="PE",
        instrument="SIM|NIFTY:24000:PE",
        price=premium,
        spot=spot,
        index_beta=-1.0,
        signals=signals(-0.6),
        projections=projection(spot, bps),
        index_projections=projection(spot, bps),
        atr=4.0,
        micro=-0.2,
        greeks={"delta": delta, "premium": premium, "theta_per_minute": 0.0},
    )


def simulator(**overrides) -> MoneySimulator:
    config = SimulatorConfig(
        opening_balance=25_000.0,
        reserve=200_000.0,
        lot_size=LOT,
        **overrides,
    )
    return MoneySimulator(config, cost_model=costs())


def enter(sim: MoneySimulator, legs, at: datetime, seconds: float = 1.0) -> None:
    """Decide, then fill.

    Entries are never filled at the price the decision was made on — that price
    is gone by the time the order exists — so a leg opens on the *next* print.
    Every test that expects a position therefore needs both moments.
    """
    sim.step(list(legs), at)
    sim.step(list(legs), at + timedelta(seconds=seconds))


# ------------------------------------------------------------- rupees per point


def test_a_point_on_the_index_is_worth_exactly_the_lot() -> None:
    """24,000 -> 24,020 is +1,300 rupees at 65 units, not +20 and not +30 million."""
    sim = simulator()
    enter(sim, [index_state(24_000.0)], START)
    assert sim.positions["INDEX"].entry_price == 24_000.0

    sim.step([index_state(24_020.0)], START + timedelta(seconds=5))
    position = sim.positions["INDEX"]
    assert position.unrealized(24_020.0) == pytest.approx(20.0 * LOT)


def test_a_loss_of_a_point_costs_the_same_as_a_gain() -> None:
    sim = simulator()
    enter(sim, [index_state(24_000.0)], START)
    position = sim.positions["INDEX"]
    assert position.unrealized(23_980.0) == pytest.approx(-20.0 * LOT)


def test_the_wallet_never_holds_the_notional() -> None:
    """Risk capital, not exposure: a 65-unit index lot never enters the balance."""
    sim = simulator()
    enter(sim, [index_state(24_000.0)], START)
    wallet = sim.wallets["INDEX"]
    # Only the entry cost has left; the 1.5 million of exposure has not.
    assert wallet.cash == pytest.approx(25_000.0 - sim.positions["INDEX"].entry_cost, abs=1e-6)
    assert wallet.cash > 24_000.0


# --------------------------------------------------------------------- one lot


def test_every_position_is_exactly_one_lot() -> None:
    sim = simulator()
    enter(sim, [index_state(24_000.0), call_state(120.0, 24_000.0)], START)
    assert sim.positions["INDEX"].units == LOT
    assert sim.positions["CALL"].units == LOT


# --------------------------------------------------------------------- costs


def test_costs_are_charged_on_both_legs_of_a_round_trip() -> None:
    sim = simulator()
    enter(sim, [index_state(24_000.0)], START)
    position = sim.positions["INDEX"]
    assert position.entry_cost > 0

    # A flat close: no price move, so the result is exactly the round trip.
    sim.step([index_state(24_000.0)], START + timedelta(minutes=10))
    assert "INDEX" not in sim.positions
    trade = sim.closed[-1]

    model = costs()
    notional = 24_000.0 * LOT
    buy = model.per_leg_bps(notional, "buy") / 10_000.0 * notional
    sell = model.per_leg_bps(notional, "sell") / 10_000.0 * notional

    assert trade.gross == pytest.approx(0.0)
    assert trade.costs == pytest.approx(buy + sell, rel=1e-6)
    assert trade.net == pytest.approx(-(buy + sell), rel=1e-6)
    # The exit leg carries STT and the entry leg carries stamp duty, so the two
    # sides of the same round trip are deliberately not the same price.
    assert sell > buy > 0
    assert trade.costs == pytest.approx(position.entry_cost + sell, rel=1e-6)


def test_an_option_is_charged_the_option_rate_not_the_futures_rate() -> None:
    """A premium's cost is a fraction of the premium, not basis points of notional."""
    sim = simulator()
    enter(sim, [call_state(120.0, 24_000.0)], START)
    trade_cost = sim.positions["CALL"].entry_cost
    # Half of the 1.2% round trip on 120 x 65 = 7,800.
    assert trade_cost == pytest.approx(7_800.0 * 0.012 / 2.0, rel=1e-6)


# --------------------------------------------------------------------- exits


def test_a_stop_fills_at_its_level_not_at_the_next_print() -> None:
    """A gap between refreshes must not turn a 20-point stop into a rout."""
    sim = simulator()
    enter(sim, [index_state(24_000.0, atr=20.0)], START)
    stop = sim.positions["INDEX"].stop_price
    assert stop == pytest.approx(24_000.0 - 1.5 * 20.0)

    # The tape is 300 points lower by the next refresh.
    sim.step([index_state(23_700.0, atr=20.0)], START + timedelta(seconds=5))
    trade = sim.closed[-1]
    assert trade.exit_reason == "stop"
    assert trade.exit_price == pytest.approx(stop)
    assert trade.gross == pytest.approx(-30.0 * LOT)


def test_a_target_closes_the_lot_for_a_profit() -> None:
    sim = simulator()
    enter(sim, [index_state(24_000.0, atr=20.0)], START)
    target = sim.positions["INDEX"].target_price
    assert target == pytest.approx(24_000.0 + 2.5 * 20.0)

    sim.step([index_state(24_060.0, atr=20.0)], START + timedelta(minutes=1))
    assert sim.closed[-1].exit_reason == "target"
    assert sim.closed[-1].net > 0


def test_a_position_is_closed_when_the_session_ends() -> None:
    sim = simulator()
    enter(sim, [index_state(24_000.0)], START)
    sim.step([index_state(24_000.0)], START + timedelta(minutes=1), session_open=False)
    assert not sim.positions
    assert sim.closed[-1].exit_reason == "session closed"


def test_entries_wait_for_the_decision_clock_but_exits_do_not() -> None:
    sim = simulator(decision_seconds=30.0, cooldown_seconds=0.0)
    enter(sim, [index_state(24_000.0, atr=20.0)], START)
    assert "INDEX" in sim.positions

    # Ten seconds on: not a decision moment, but a stop is a price level.
    sim.step([index_state(23_700.0, atr=20.0)], START + timedelta(seconds=10))
    assert "INDEX" not in sim.positions
    assert sim.closed[-1].exit_reason == "stop"


# ------------------------------------------------------- every tick matters


def test_a_stop_is_tested_on_every_print_not_once_a_second() -> None:
    """A level the tape crosses and recovers from between refreshes still counts."""
    sim = simulator()
    enter(sim, [index_state(24_000.0, atr=20.0)], START)
    stop = sim.positions["INDEX"].stop_price

    # One print pokes through the stop. Nothing else in this test runs on a clock.
    sim.mark({"INDEX": stop - 1.0}, START + timedelta(seconds=2))

    assert "INDEX" not in sim.positions, "the stop was not tested on the tick"
    assert sim.closed[-1].exit_reason == "stop"


def test_marking_a_tick_does_not_open_a_position() -> None:
    """Risk runs on every print; entries stay on the decision clock."""
    sim = simulator()
    for step in range(50):
        sim.mark({"INDEX": 24_000.0 + step}, START + timedelta(seconds=step))
    assert not sim.positions
    assert not sim._intents
    assert sim.steps == 0, "a tick is not a decision"


def test_a_target_is_also_visible_to_the_tick_path() -> None:
    sim = simulator()
    enter(sim, [index_state(24_000.0, atr=20.0)], START)
    target = sim.positions["INDEX"].target_price

    sim.mark({"INDEX": target + 5.0}, START + timedelta(seconds=2))
    assert sim.closed[-1].exit_reason == "target"


def test_the_decision_clock_matches_the_configured_cadence() -> None:
    """Ten seconds by default, and the clock is what it says it is."""
    sim = simulator()
    assert sim.config.decision_seconds == 10.0, "the cadence moved off 30s on purpose"

    sim.step([index_state(24_000.0, atr=20.0)], START)
    first = sim.last_decision_at
    assert first == START

    # Nine seconds later is not a decision moment.
    sim.step([index_state(24_000.0, atr=20.0)], START + timedelta(seconds=9))
    assert sim.last_decision_at == first

    # Eleven is.
    later = START + timedelta(seconds=11)
    sim.step([index_state(24_000.0, atr=20.0)], later)
    assert sim.last_decision_at == later


def test_an_entry_waits_for_a_price_the_tape_prints() -> None:
    """The price a decision was made on is gone by the time the order exists."""
    sim = simulator()
    sim.step([index_state(24_000.0)], START)

    assert not sim.positions, "a position was opened at the decision price"
    assert "INDEX" in sim._intents

    # The next print fills it, and it fills there — not where the view was taken.
    sim.step([index_state(24_005.0)], START + timedelta(seconds=1))
    assert sim.positions["INDEX"].entry_price == 24_005.0


def test_an_order_with_nothing_to_fill_against_is_pulled() -> None:
    """A view is about the next few minutes; it does not keep forever."""
    sim = simulator()
    sim.step([index_state(24_000.0)], START)
    assert "INDEX" in sim._intents

    sim.step([index_state(0.0)], START + timedelta(seconds=1))
    stale = START + timedelta(seconds=sim.config.entry_timeout_seconds + 1)
    sim.step([index_state(0.0)], stale)

    assert not sim._intents
    assert not sim.positions
    assert any("pulled" in decision.reason for decision in sim.decisions)


# -------------------------------------------------- the forward path decides


def test_a_favourable_tick_alone_does_not_open_a_position() -> None:
    """The complaint this exists to answer: a print went the right way, so it bought.

    A tick with no projected path behind it has nothing to justify a position —
    and momentum carries no weight, so it cannot be the reason either.
    """
    sim = simulator()
    state = index_state(24_000.0)
    state.projections = ()
    state.index_projections = ()
    state.micro = 1.0
    state.signals = signals(1.0)

    sim.step([state], START)
    assert not sim._intents
    assert not sim.positions

    action, reason = sim.policy.action_for(sim.views["INDEX"], has_position=False)
    assert action == HOLD
    assert "no projected path" in reason


def test_an_entry_needs_the_projected_path_to_lead_the_view() -> None:
    """Strong rules are not enough on their own; the forward view has to carry it."""
    policy = DecisionPolicy(units=LOT)
    state = index_state(24_000.0, bps=0.3)
    state.signals = signals(1.0)

    view = policy.view_for(state, costs())
    assert view.leading_source != "projection"
    action, reason = policy.action_for(view, has_position=False)
    assert action == HOLD
    assert "does not lead" in reason


def test_an_entry_is_refused_when_the_path_bends_back() -> None:
    """A forecast that turns against the trade inside its own horizon is not one.

    Whichever side the view ends up taking, the bar that points the other way is
    the forecast saying the position should be closed before its target — so there
    is no reason to open it.
    """
    policy = DecisionPolicy(units=LOT)
    state = index_state(24_000.0)
    up = projection(24_000.0, 8.0 / 3)
    down = projection(up[-1].close, -40.0)
    state.index_projections = (up[0], up[1], down[0])

    view = policy.view_for(state, costs())
    assert view.horizon_directions == (1, 1, -1)
    action, reason = policy.action_for(view, has_position=False)
    assert action == HOLD
    assert "disagrees" in reason
    # And the reason names the bars that disagree, so a reader can see which.
    assert "+1m" in reason and "+2m" in reason


def test_every_bar_of_the_path_must_point_where_the_trade_does() -> None:
    policy = DecisionPolicy(units=LOT)
    view = policy.view_for(index_state(24_000.0), costs())
    assert view.horizon_directions == (1, 1, 1)
    assert policy.action_for(view, has_position=False)[0] == BUY

    falling = policy.view_for(index_state(24_000.0, bps=-8.0), costs())
    assert falling.horizon_directions == (-1, -1, -1)
    assert policy.action_for(falling, has_position=False)[0] == SELL


def test_a_puts_path_is_read_in_the_puts_own_direction() -> None:
    """The projected path is the index's, so it flips with the leg's sign."""
    policy = DecisionPolicy(units=LOT)
    view = policy.view_for(put_state(120.0, 24_000.0), costs())
    assert view.view > 0
    assert view.horizon_directions == (1, 1, 1), "a long put follows a falling index"
    assert policy.action_for(view, has_position=False)[0] == BUY


def test_zero_weight_momentum_does_not_count_as_confirmation() -> None:
    """Switching a source off has to remove its vote, not just its arithmetic."""
    policy = DecisionPolicy(units=LOT, weights=PolicyWeights(momentum=0.0))
    view = policy.view_for(index_state(24_000.0), costs())
    _, total = view.agreement
    assert total == 2, "a weighted-out source is not an opinion"
    assert "momentum" not in [item.source for item in view.contributions if item.weight > 0]


# --------------------------------------------------------------------- policy


def test_a_bearish_index_turns_into_a_long_put() -> None:
    sim = simulator()
    enter(sim, [put_state(120.0, 24_000.0)], START)
    assert "PUT" in sim.positions
    assert sim.positions["PUT"].direction is Direction.UP


def test_a_put_agrees_with_the_sources_that_agree_with_it() -> None:
    """A put's own price is anti-correlated with the index, and that is not dissent."""
    policy = DecisionPolicy(units=LOT)
    view = policy.view_for(put_state(120.0, 24_000.0), costs())
    assert view.view > 0, "a bearish index view must be a bullish premium view for a put"
    agree, total = view.agreement
    assert agree == total == 2


def test_an_option_hurdle_includes_the_decay_it_pays_while_held() -> None:
    """Spread alone quotes a hurdle a position clears by simply refusing to wait."""
    policy = DecisionPolicy(units=LOT, horizon_bars=3)
    cheaper = policy.required_move_bps(call_state(120.0, 24_000.0), costs())

    state = call_state(120.0, 24_000.0)
    state.greeks["theta_per_minute"] = -0.5
    dearer = policy.required_move_bps(state, costs())

    assert dearer > cheaper
    assert dearer == pytest.approx(cheaper + 0.5 * 3 / 120.0 * 10_000.0, rel=1e-6)


def test_a_weak_view_holds_with_its_reason() -> None:
    policy = DecisionPolicy(units=LOT, entry_view=0.5)
    view = policy.view_for(index_state(24_000.0, bps=0.2), costs())
    action, reason = policy.action_for(view, has_position=False)
    assert action == HOLD
    assert "below" in reason


def test_a_missing_source_does_not_dilute_the_others() -> None:
    """Offline, with no trained model, the remaining sources read as a full opinion."""
    policy = DecisionPolicy(units=LOT, weights=PolicyWeights())
    without = policy.view_for(index_state(24_000.0), costs())

    with_model = index_state(24_000.0)
    with_model.predictions = (type("P", (), {"p_up": 0.5})(),)
    blended = policy.view_for(with_model, costs())

    # A model with no edge must not move the view towards zero by its weight alone.
    assert blended.view == pytest.approx(without.view, abs=1e-6)


def test_the_projection_is_read_in_the_underlying_not_in_the_leg() -> None:
    """A premium's own path is levered; reading it as the index overstates by ~65x."""
    policy = DecisionPolicy(units=LOT)
    state = call_state(120.0, 24_000.0)
    # The leg's own projection climbs 170bp a bar; the hurdle must ignore it.
    state.projections = projection(120.0, 510.0)
    assert policy.projected_index_move_bps(state) == pytest.approx(8.0, rel=2e-3)

    view = policy.view_for(state, costs())
    # 8bp of index at 0.5 delta on a 24,000 spot is 9.6 points of premium, which
    # on a 120-rupee premium is 800bp — not the 510bp the premium's own path said.
    assert view.expected_move_bps == pytest.approx(9.6 / 120.0 * 10_000.0, rel=2e-3)


def test_an_entry_needs_the_round_trip_to_be_paid_for() -> None:
    """min_edge_bps is enforced, not decorative."""
    tight = DecisionPolicy(units=LOT, min_edge_bps=0.0)
    greedy = DecisionPolicy(units=LOT, min_edge_bps=10_000.0)
    view = tight.view_for(index_state(24_000.0), costs())
    assert tight.action_for(view, has_position=False)[0] == BUY
    assert greedy.action_for(view, has_position=False)[0] == HOLD


# ------------------------------------------------------------------- reserve


def test_a_drawn_down_wallet_borrows_and_trades_again() -> None:
    sim = simulator(cooldown_seconds=0.0)
    enter(sim, [index_state(24_000.0, atr=150.0)], START)
    sim.step([index_state(23_600.0, atr=150.0)], START + timedelta(seconds=5))

    wallet = sim.wallets["INDEX"]
    assert wallet.cash < 12_500.0, "the loss must genuinely exhaust the leg"
    assert sim.reserve.deployed == pytest.approx(0.0)

    # The top-up happens where the lot is funded, which is the fill and not the
    # judgement — nothing is bought until a price prints against the order.
    enter(sim, [index_state(23_600.0, atr=150.0)], START + timedelta(minutes=1))
    assert wallet.reserve_drawn > 0
    assert sim.reserve.remaining < 200_000.0
    assert sim.positions["INDEX"], "a topped-up leg must be able to invest again"


def test_the_reserve_is_repaid_before_profits_belong_to_the_leg() -> None:
    reserve = Reserve(opening=200_000.0)
    wallet = Wallet(label="INDEX", opening=25_000.0, cash=10_000.0)

    lent = reserve.lend("INDEX", 15_000.0)
    wallet.cash += lent
    wallet.reserve_drawn += lent
    assert wallet.cash == 25_000.0
    assert reserve.remaining == 185_000.0

    sim = simulator()
    sim.wallets["INDEX"] = wallet
    sim.reserve = reserve
    wallet.settle(cash_delta=2_000.0, net_pnl=2_000.0, costs=0.0)
    sim._sweep("INDEX")

    assert wallet.cash == pytest.approx(25_000.0), "profit above the opening is not the leg's"
    assert wallet.reserve_drawn == pytest.approx(13_000.0)
    assert reserve.remaining == pytest.approx(187_000.0)


def test_the_reserve_runs_out_rather_than_going_negative() -> None:
    reserve = Reserve(opening=1_000.0)
    assert reserve.lend("INDEX", 5_000.0) == 1_000.0
    assert reserve.remaining == 0.0
    assert reserve.lend("CALL", 5_000.0) == 0.0
    assert reserve.remaining == 0.0


# ------------------------------------------------------------------- ledger


def test_wallets_survive_a_restart(tmp_path) -> None:
    path = tmp_path / "money.sqlite3"
    first = MoneySimulator(
        SimulatorConfig(opening_balance=25_000.0, reserve=200_000.0, lot_size=LOT),
        ledger=SimulationLedger(path, "simulation"),
        cost_model=costs(),
    )
    enter(first, [index_state(24_000.0)], START)
    first.step([index_state(24_000.0)], START + timedelta(minutes=10))
    expected = first.wallets["INDEX"].cash
    assert first.closed

    revived = MoneySimulator(
        SimulatorConfig(opening_balance=25_000.0, reserve=200_000.0, lot_size=LOT),
        ledger=SimulationLedger(path, "simulation"),
        cost_model=costs(),
    )
    assert revived.wallets["INDEX"].cash == pytest.approx(expected)
    assert revived.wallets["INDEX"].trades == 1


def test_live_and_simulation_books_are_separate(tmp_path) -> None:
    money = tmp_path / "money.sqlite3"
    live = SimulationLedger(money, "live")
    paper = SimulationLedger(money, "simulation")

    sim = MoneySimulator(SimulatorConfig(), ledger=paper, cost_model=costs())
    enter(sim, [index_state(24_000.0)], START)
    sim.step([index_state(24_000.0)], START + timedelta(minutes=10))

    assert paper.totals()["trades"] == 1
    assert live.totals()["trades"] == 0


def test_a_reset_hands_every_leg_its_opening_balance_back(tmp_path) -> None:
    sim = MoneySimulator(
        SimulatorConfig(opening_balance=25_000.0, reserve=200_000.0, lot_size=LOT),
        ledger=SimulationLedger(tmp_path / "money.sqlite3", "simulation"),
        cost_model=costs(),
    )
    enter(sim, [index_state(24_000.0)], START)
    sim.step([index_state(24_000.0)], START + timedelta(minutes=10))
    assert sim.total_pnl() != 0

    sim.reset()
    assert sim.total_pnl() == 0
    assert sim.positions == {}
    assert sim.reserve.remaining == 200_000.0
    assert sim.snapshot()["wallets"] == []


# --------------------------------------------------------------- presentation


def test_the_snapshot_is_json_safe_and_carries_the_policy() -> None:
    import json

    sim = simulator()
    enter(sim, [index_state(24_000.0), call_state(120.0, 24_000.0)], START)
    payload = json.loads(json.dumps(sim.snapshot()))

    assert payload["lots"] == LOT
    assert payload["policy"]["lot_size"] == LOT
    assert payload["policy"]["opening_balance"] == 25_000.0
    assert {wallet["label"] for wallet in payload["wallets"]} == {"INDEX", "CALL"}
    index = next(item for item in payload["wallets"] if item["label"] == "INDEX")
    assert index["position"]["units"] == LOT
    assert index["view"]["sources"], "the reason behind the view must travel with it"


# ----------------------------------------------------------------- board wiring


@pytest.fixture(scope="module")
def board(tmp_path_factory):
    from kernel import Kernel
    from plugins.sources.simulated.series import generate_candles
    from runtime.board import MarketBoard

    # A temporary data directory, not the project's. The board builds the paper
    # book from the capability, the book persists after every close, and an
    # offline dashboard reads the same ``simulation`` ledger — so a test that
    # traded against the real data directory would write its own fake fills into
    # the user's paper book.
    root = tmp_path_factory.mktemp("money-board")
    settings = Settings(data_dir=root / "data", model_dir=root / "models", log_dir=root / "logs")
    return MarketBoard(Kernel.bootstrap(settings), bar_minutes=1).build(
        generate_candles(days=3, seed=29)
    )


def test_the_board_reaches_the_book_through_the_capability(board) -> None:
    assert board.simulator is not None
    assert board.kernel.registry.capabilities()["money_simulator"] == "advisory:money_simulator"


def test_the_board_gives_every_leg_its_sign_against_the_index(board) -> None:
    """A put is short the index, and the payload has to say so."""
    payloads = {leg.label: board._leg_payload(leg) for leg in board.legs}
    assert payloads["INDEX"]["index_beta"] == 1.0
    assert payloads["CALL"]["index_beta"] == 1.0
    assert payloads["PUT"]["index_beta"] == -1.0


def test_every_leg_reads_the_underlyings_path(board) -> None:
    """An option's own projection is a levered premium series, not the index's."""
    payloads = {leg.label: board._leg_payload(leg) for leg in board.legs}
    underlying = payloads["INDEX"]["index_projections"]
    assert payloads["CALL"]["index_projections"] == underlying
    assert payloads["PUT"]["index_projections"] == underlying
    assert payloads["INDEX"]["index_projections"] == payloads["INDEX"]["projections"]


def test_the_board_does_not_trade_before_anything_has_printed(board) -> None:
    """A decision on a warm-up frame is a decision on a price nobody could trade."""
    assert board.index_engine.last_tick_ts is None
    board.refresh_projection()
    assert board.simulator.steps == 0


def test_a_board_tick_moves_the_paper_book(board) -> None:
    from datetime import datetime

    from core.calendar import IST
    from core.types import Tick

    before = board.simulator.steps
    board.on_tick(Tick(ts=datetime.now(IST), ltp=board.index_engine.last_price))
    board.refresh_projection()
    assert board.simulator.steps > before

    snapshot = board.snapshot()
    assert snapshot.simulation["wallets"], "the book must be on the frame the renderer draws"


def test_the_renderer_receives_the_paper_book(board) -> None:
    from plugins.renderers.web.serialize import serialize_snapshot

    payload = serialize_snapshot(board.snapshot(), status="SIMULATION")
    assert "simulation" in payload
    assert payload["simulation"]["lots"] == 65
    assert payload["simulation"]["reserve"]["opening"] == 200_000.0
    assert payload["simulation"]["policy"]["lot_size"] == 65
