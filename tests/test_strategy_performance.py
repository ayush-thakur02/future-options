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
