from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from core.settings import Settings
from kernel import Kernel
from plugins.forecasts.online_research import OnlineResearchLab, ResearchAction
from plugins.forecasts.online_research.algorithms import ALGORITHMS
from plugins.forecasts.online_research.metrics import conservative_trust, summarise
from plugins.forecasts.online_research.types import PredictionRecord

NOW = datetime(2026, 9, 14, 9, 15, tzinfo=UTC)


def test_plugin_is_discovered_as_an_independent_capability(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path / "data", model_dir=tmp_path / "models")
    kernel = Kernel.bootstrap(settings, with_entry_points=False)

    assert kernel.provider("online_research") == "forecast:online_research"
    lab = kernel.build(
        "forecast:online_research",
        state_path=tmp_path / "state.sqlite3",
        algorithms=("online_logistic",),
    )
    assert isinstance(lab, OnlineResearchLab)
    assert kernel.provider("forecast") == "forecast:ml_ensemble"


def test_models_do_not_update_before_a_prediction_matures(tmp_path) -> None:
    path = tmp_path / "online.sqlite3"
    lab = OnlineResearchLab(path, algorithms=("online_logistic",))

    signal = lab.issue(
        {"momentum": 1.5, "noise": float("nan")},
        timestamp=NOW,
        anchor_price=100.0,
        horizon_min=5,
        instrument="TEST",
    )
    assert signal.label == "RESEARCH SIGNAL: HOLD"
    assert signal.predictions[0].p_up == pytest.approx(0.5)
    assert lab.algorithms["online_logistic"].state_dict()["samples_seen"] == 0
    assert lab.observe(timestamp=NOW + timedelta(minutes=4), price=101.0, instrument="TEST") == []
    assert lab.scorecards()[0].all_time.sample_count == 0

    matured = lab.observe(timestamp=NOW + timedelta(minutes=5), price=101.0, instrument="TEST")
    assert len(matured) == 1
    assert matured[0].label_up == 1
    assert matured[0].hit is True
    assert matured[0].brier == pytest.approx(0.25)
    assert lab.algorithms["online_logistic"].state_dict()["samples_seen"] == 1
    with sqlite3.connect(path) as database:
        persisted = database.execute(
            "SELECT scored, payload_json FROM prediction_ledger"
        ).fetchone()
    assert persisted is not None and persisted[0] == 1


def test_overlapping_forecasts_are_frozen_before_the_first_outcome(tmp_path) -> None:
    lab = OnlineResearchLab(tmp_path / "online.sqlite3", algorithms=("online_logistic",))
    first = lab.issue(
        {"momentum": 1.0},
        timestamp=NOW,
        anchor_price=100.0,
        horizon_min=2,
        instrument="TEST",
    )
    second = lab.issue(
        {"momentum": 1.0},
        timestamp=NOW + timedelta(minutes=1),
        anchor_price=100.0,
        horizon_min=2,
        instrument="TEST",
    )

    assert first.p_up == pytest.approx(second.p_up)
    assert lab.algorithms["online_logistic"].state_dict()["samples_seen"] == 0
    lab.observe(timestamp=NOW + timedelta(minutes=2), price=101.0, instrument="TEST")
    assert lab.algorithms["online_logistic"].state_dict()["samples_seen"] == 1
    assert lab.pending_count == 1


def test_every_algorithm_gets_its_own_prediction_and_outcome_record(tmp_path) -> None:
    lab = OnlineResearchLab(tmp_path / "online.sqlite3", min_trust_for_action=0.0)
    signal = lab.issue(
        {"return_1": -0.01, "rsi": 32.0},
        timestamp=NOW,
        anchor_price=20_000.0,
        horizon_min=1,
        instrument="INDEX",
        cost_bps=1.5,
    )

    assert {item.algorithm for item in signal.predictions} == set(ALGORITHMS)
    assert len({item.prediction_id for item in signal.predictions}) == len(ALGORITHMS)
    assert lab.pending_count == len(ALGORITHMS)

    matured = lab.observe(timestamp=NOW + timedelta(minutes=1), price=19_980.0, instrument="INDEX")
    assert len(matured) == len(ALGORITHMS)
    assert all(record.actual_return_bps == pytest.approx(-10.0) for record in matured)
    assert all(record.label_up == 0 for record in matured)
    assert all(record.gross_pnl_bps == 0.0 for record in matured)  # initial P(up)=0.5 => HOLD
    assert {card.algorithm for card in lab.scorecards()} == set(ALGORITHMS)
    assert all(card.all_time.sample_count == 1 for card in lab.scorecards())


@pytest.mark.parametrize("algorithm", tuple(ALGORITHMS))
def test_pending_ledger_and_fitted_models_survive_restart(tmp_path, algorithm) -> None:
    path = tmp_path / "online.sqlite3"
    first = OnlineResearchLab(path, algorithms=(algorithm,))
    first.issue(
        {"momentum": 2.0},
        timestamp=NOW,
        anchor_price=100.0,
        horizon_min=2,
        instrument="TEST",
    )

    restarted = OnlineResearchLab(path, algorithms=(algorithm,))
    assert restarted.pending_count == 1
    restarted.observe(timestamp=NOW + timedelta(minutes=2), price=101.0, instrument="TEST")

    second_restart = OnlineResearchLab(path, algorithms=(algorithm,))
    assert second_restart.pending_count == 0
    assert second_restart.scorecards()[0].all_time.sample_count == 1
    assert second_restart.algorithms[algorithm].state_dict()["samples_seen"] == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_observations_only_mature_the_matching_instrument(tmp_path) -> None:
    lab = OnlineResearchLab(tmp_path / "online.sqlite3", algorithms=("online_logistic",))
    for instrument in ("A", "B"):
        lab.issue(
            {"feature": 1.0},
            timestamp=NOW,
            anchor_price=100.0,
            horizon_min=1,
            instrument=instrument,
        )

    matured = lab.observe(timestamp=NOW + timedelta(minutes=1), price=102.0, instrument="A")
    assert len(matured) == 1
    assert matured[0].instrument == "A"
    assert lab.pending_count == 1


def test_scorecard_reports_calibration_profit_loss_and_conservative_trust() -> None:
    records = []
    for index in range(100):
        target = index % 2
        probability = 0.9 if target else 0.1
        pnl = 8.0 if index != 99 else -3.0
        records.append(
            PredictionRecord(
                prediction_id=str(index),
                algorithm="oracle",
                instrument="TEST",
                issued_at=NOW + timedelta(minutes=index),
                target_at=NOW + timedelta(minutes=index + 1),
                horizon_min=1,
                anchor_price=100.0,
                p_up=probability,
                action=ResearchAction.BUY if target else ResearchAction.SELL,
                confidence=0.8,
                trust_at_issue=0.0,
                cost_bps=1.0,
                features={"x": float(index)},
                matured_at=NOW + timedelta(minutes=index + 1),
                actual_price=101.0 if target else 99.0,
                actual_return_bps=100.0 if target else -100.0,
                label_up=target,
                hit=True,
                brier=0.01,
                gross_pnl_bps=pnl + 1.0,
                net_pnl_bps=pnl,
            )
        )

    summary = summarise(records)
    assert summary.sample_count == 100
    assert summary.accuracy == 1.0
    assert summary.brier_score == pytest.approx(0.01)
    assert summary.calibration_error == pytest.approx(0.1)
    assert summary.trade_count == 100
    assert summary.profit_bps == pytest.approx(99 * 8.0)
    assert summary.loss_bps == pytest.approx(3.0)
    assert summary.net_pnl_bps == pytest.approx(99 * 8.0 - 3.0)
    assert 0.0 < conservative_trust(summary, min_samples=50) < 1.0


def test_online_logistic_changes_only_after_repeated_matured_outcomes(tmp_path) -> None:
    lab = OnlineResearchLab(
        tmp_path / "online.sqlite3",
        algorithms=("online_logistic",),
        min_trust_for_action=0.0,
    )
    first_probability = None
    last_probability = None
    for index in range(30):
        issued = NOW + timedelta(minutes=index * 2)
        signal = lab.issue(
            {"positive_regime": 1.0},
            timestamp=issued,
            anchor_price=100.0,
            horizon_min=1,
            instrument="TEST",
        )
        first_probability = signal.p_up if first_probability is None else first_probability
        last_probability = signal.p_up
        lab.observe(timestamp=issued + timedelta(minutes=1), price=101.0, instrument="TEST")

    assert first_probability == pytest.approx(0.5)
    assert last_probability is not None and last_probability > 0.60
    assert signal.action == ResearchAction.BUY
    assert lab.scorecards()[0].all_time.sample_count == 30
    assert lab.records(limit=1)[0].net_pnl_bps == pytest.approx(100.0)


def test_scorecard_keeps_a_distinct_rolling_window(tmp_path) -> None:
    lab = OnlineResearchLab(
        tmp_path / "online.sqlite3",
        algorithms=("online_logistic",),
        rolling_window=2,
    )
    for index in range(3):
        issued = NOW + timedelta(minutes=index * 2)
        lab.issue(
            {"x": float(index)},
            timestamp=issued,
            anchor_price=100.0,
            horizon_min=1,
            instrument="TEST",
        )
        lab.observe(timestamp=issued + timedelta(minutes=1), price=101.0, instrument="TEST")

    card = lab.scorecards()[0]
    assert card.all_time.sample_count == 3
    assert card.rolling.sample_count == 2


def test_invalid_inputs_never_enter_the_persistent_ledger(tmp_path) -> None:
    lab = OnlineResearchLab(tmp_path / "online.sqlite3")
    with pytest.raises(ValueError, match="positive"):
        lab.issue({}, timestamp=NOW, anchor_price=0.0, horizon_min=1)
    with pytest.raises(ValueError, match="finite numeric"):
        lab.issue({"bad": float("nan")}, timestamp=NOW, anchor_price=100.0, horizon_min=1)
    assert not (tmp_path / "online.sqlite3").exists()
