from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from core.settings import Settings
from core.types import Direction, Signal
from kernel import Kernel
from plugins.advisory.performance_ledger import StrategyPerformanceLedger

NOW = datetime(2026, 9, 14, 9, 15, tzinfo=UTC)


def _signal(name: str, direction: Direction = Direction.UP) -> Signal:
    return Signal(NOW, name, direction, 0.8, "test")


def test_performance_ledger_is_a_discovered_capability(tmp_path) -> None:
    kernel = Kernel.bootstrap(Settings(data_dir=tmp_path), with_entry_points=False)
    assert kernel.provider("strategy_performance") == "advisory:performance_ledger"


def test_active_strategy_outcomes_and_costs_survive_restart(tmp_path) -> None:
    path = tmp_path / "strategies.sqlite3"
    ledger = StrategyPerformanceLedger(path, min_trust_samples=1)
    assert ledger.issue(
        [_signal("trend"), _signal("fade", Direction.DOWN)],
        timestamp=NOW,
        anchor_price=100.0,
        horizon_min=1,
        instrument="TEST",
        cost_bps=2.0,
    ) == 2

    outcomes = ledger.observe(
        timestamp=NOW + timedelta(minutes=1), price=101.0, instrument="TEST"
    )
    assert len(outcomes) == 2
    cards = {card.strategy: card for card in StrategyPerformanceLedger(path).scorecards()}
    assert cards["trend"].accuracy == 1.0
    assert cards["trend"].net_pnl_bps == pytest.approx(98.0)
    assert cards["fade"].loss_bps == pytest.approx(102.0)


def test_missed_target_is_expired_instead_of_scored_at_a_later_price(tmp_path) -> None:
    ledger = StrategyPerformanceLedger(tmp_path / "strategies.sqlite3")
    ledger.issue(
        [_signal("trend")],
        timestamp=NOW,
        anchor_price=100.0,
        horizon_min=1,
        instrument="TEST",
        cost_bps=1.0,
    )
    outcomes = ledger.observe(
        timestamp=NOW + timedelta(minutes=2), price=105.0, instrument="TEST"
    )
    assert outcomes == []
    assert ledger.counts() == {"expired": 1}
    assert ledger.scorecards() == []


def test_flat_wait_signals_are_not_counted_as_trades(tmp_path) -> None:
    ledger = StrategyPerformanceLedger(tmp_path / "strategies.sqlite3")
    assert ledger.issue(
        [_signal("waiting", Direction.FLAT)],
        timestamp=NOW,
        anchor_price=100.0,
        horizon_min=1,
        instrument="TEST",
        cost_bps=1.0,
    ) == 0
    assert ledger.counts() == {}


# --------------------------------------------------------------- signed trust


def _trust(hits: int, samples: int, net_pnl_sum: float = 0.0) -> float:
    from core.scoring import conservative_strategy_trust

    return conservative_strategy_trust(
        hits=hits, samples=samples, net_pnl_sum=net_pnl_sum, min_trust_samples=50
    )


def test_a_reliably_wrong_rule_scores_negative() -> None:
    """A rule that loses can be counted on too, and that is worth a number.

    Clamping at zero would make "no evidence" and "evidence it is wrong"
    indistinguishable, and a consumer could only ever weight such a rule down to
    the same place as one nobody has measured.
    """
    wrong = _trust(hits=100, samples=500)  # 20% over 500 outcomes
    assert wrong < 0.0
    right = _trust(hits=400, samples=500)  # 80% over the same
    assert right > 0.0


def test_the_same_rate_over_a_small_sample_claims_nothing() -> None:
    """Five outcomes pointing the wrong way is not evidence of anything."""
    assert _trust(hits=1, samples=5) == pytest.approx(0.0, abs=1e-9)
    assert _trust(hits=4, samples=5) == pytest.approx(0.0, abs=1e-9)


def test_a_coin_flip_earns_nothing_either_way() -> None:
    assert _trust(hits=250, samples=500) == pytest.approx(0.0, abs=1e-9)


def test_an_accurate_rule_that_loses_money_still_scores_down() -> None:
    """Accuracy is not the question; the post-cost result is."""
    profitable = _trust(hits=320, samples=500, net_pnl_sum=5_000.0)
    bleeding = _trust(hits=320, samples=500, net_pnl_sum=-5_000.0)
    assert profitable > 0.0 > bleeding
    assert profitable > abs(bleeding)


def test_trust_is_bounded_to_one_either_way() -> None:
    """A confidence interval never quite reaches the extremes, and never passes them."""
    perfect = _trust(hits=10_000, samples=10_000, net_pnl_sum=1e9)
    hopeless = _trust(hits=0, samples=10_000, net_pnl_sum=-1e9)
    assert 0.99 < perfect <= 1.0
    assert -1.0 <= hopeless < -0.99


def test_trust_grows_with_the_sample() -> None:
    """The same hit rate is worth more the more outcomes stand behind it."""
    thin = _trust(hits=35, samples=50)
    thick = _trust(hits=350, samples=500)
    assert 0.0 < thin < thick


def test_no_outcomes_means_no_claim() -> None:
    assert _trust(hits=0, samples=0) == 0.0


def test_the_weight_a_rule_carries_never_inverts_it() -> None:
    """Down-weighting a discredited rule is a smaller claim than flipping it."""
    from core.scoring import trust_weight

    assert trust_weight(0.0) == pytest.approx(0.75)
    assert trust_weight(1.0) == pytest.approx(1.25)
    assert trust_weight(-1.0) == pytest.approx(0.25)
    assert all(trust_weight(value) > 0 for value in (-1.0, -0.5, 0.0, 0.5, 1.0))
    # Out-of-range input is clamped rather than extrapolated.
    assert trust_weight(-9.0) == pytest.approx(0.25)
    assert trust_weight(9.0) == pytest.approx(1.25)
