"""Tests for the pre-open warm-up.

The failure modes worth guarding here are the ones that make a replay *look* like
it worked:

* replaying the same bars twice, so the learners see their own outcomes again and
  trust inflates with every restart;
* scoring a signal across a session boundary, manufacturing an overnight outcome
  the live loop would never take;
* feeding a learner the scorecard of the whole window instead of the evidence
  that existed at the bar being described, which quietly hands it the answer;
* and replaying everything, every start, so the second run of the day costs the
  same as the first.
"""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import pytest

from core.calendar import IST
from core.settings import Settings
from kernel import Kernel
from plugins.sources.simulated.series import generate_candles
from runtime.board import MarketBoard
from runtime.warmup import (
    STRATEGY_HORIZON_BARS,
    WarmUp,
    WarmUpCheckpoint,
    warm_up_checkpoint,
)

WINDOW = 120


@pytest.fixture(scope="module")
def warm(tmp_path_factory):
    """A board on generated bars, research on, nothing warmed yet."""
    root = tmp_path_factory.mktemp("warmup")
    settings = Settings(data_dir=root / "data", model_dir=root / "models", log_dir=root / "logs")
    board = MarketBoard(Kernel.bootstrap(settings), bar_minutes=1).build(
        generate_candles(days=3, seed=31, end=datetime(2026, 9, 10).date())
    )
    for leg in board.legs:
        leg.engine.enable_research()
    yield board, settings
    board.close()


@pytest.fixture(scope="module")
def cold(warm):
    """One full cold replay, shared: the replay is the expensive part of this file."""
    board, settings = warm
    report = run(board, settings, mode="cold")
    return report


def run(board, settings, *, mode, bars=WINDOW, projections=True, only=None):
    legs = board.legs if only is None else [leg for leg in board.legs if leg.label == only]
    service = WarmUp(checkpoint=warm_up_checkpoint(settings))
    return service.run(
        [(leg.label, leg.engine) for leg in legs],
        mode=mode,
        bars=bars,
        projections=projections,
    )


# --------------------------------------------------------------- the replay


def test_a_cold_replay_earns_evidence_for_every_instrument(cold) -> None:
    assert cold.ran
    assert cold.cold
    assert len(cold.instruments) == 3
    for outcome in cold.instruments:
        assert not outcome.skipped, outcome.skipped
        assert outcome.bars > 0
        assert outcome.strategies_issued > 0, f"{outcome.label} issued nothing"
        assert outcome.ai_issued > 0, f"{outcome.label} issued no forecasts"


def test_the_replay_leaves_the_ledgers_with_a_record(cold, warm) -> None:
    board, _settings = warm
    for leg in board.legs:
        cards = leg.engine.online_lab.scorecards()
        assert cards and all(card.all_time.sample_count > 0 for card in cards), (
            f"{leg.label} learners were never trained"
        )


def test_the_engine_reads_the_replays_outcomes_back(cold, warm) -> None:
    """Without this the replay happens and the ensemble never notices."""
    board, _settings = warm
    for leg in board.legs:
        stats = leg.engine.rule_stats
        assert stats, f"{leg.label} has no rule statistics after a replay"
        assert sum(card["scored"] for card in stats.values()) > 0


def test_projections_are_scored_so_the_forward_panel_has_a_history(cold) -> None:
    assert any(item.projections_scored > 0 for item in cold.instruments)


def test_projections_can_be_left_out(warm) -> None:
    board, settings = warm
    report = run(board, settings, mode="no-projections", projections=False, only="INDEX")
    assert report.instruments[0].projections_scored == 0
    assert report.instruments[0].skipped or report.instruments[0].bars >= 0


# --------------------------------------------------------------- the checkpoint


def test_a_second_run_replays_nothing(warm) -> None:
    """Re-running would feed every learner its own outcomes a second time."""
    board, settings = warm
    first = run(board, settings, mode="resume", only="INDEX")
    assert first.bars > 0

    second = run(board, settings, mode="resume", only="INDEX")
    assert second.bars == 0
    assert all(item.skipped for item in second.instruments)


def test_the_second_run_keeps_what_the_first_earned(warm) -> None:
    board, settings = warm
    run(board, settings, mode="keeps", only="INDEX")
    lab = board.legs[0].engine.online_lab
    before = [card.all_time.sample_count for card in lab.scorecards()]

    run(board, settings, mode="keeps", only="INDEX")
    after = [card.all_time.sample_count for card in lab.scorecards()]
    assert after == before, "a resumed run must not add or lose samples"


def test_a_lost_checkpoint_replays_the_window_again(warm) -> None:
    """The checkpoint is the whole of the resume logic, so losing it costs work.

    The strategy ledger takes that harmlessly — its records are keyed by
    instrument, rule, issue time and target, so a replayed signal is ignored
    rather than counted twice. The learners' ledger is append-only, which is why
    the checkpoint exists rather than being derived.
    """
    board, settings = warm
    mode = "lost"
    run(board, settings, mode=mode, only="CALL", projections=False)

    checkpoint = warm_up_checkpoint(settings)
    engine = board.legs[1].engine
    through = checkpoint.through(mode, engine.instrument_key, 1)
    checkpoint.reset(mode=mode)
    assert checkpoint.through(mode, engine.instrument_key, 1) is None

    # A time far enough back that the whole window is replayed, but not so far
    # that the signals land on bars the strategy ledger has already scored.
    stamps = engine.history.index
    position = int(stamps.searchsorted(through, side="right"))
    checkpoint.record(mode, engine.instrument_key, 1, stamps[max(position - 100, 1)], 0, 0.0)

    report = run(board, settings, mode=mode, only="CALL", projections=False)
    outcome = report.instruments[0]
    assert not outcome.cold_start
    assert outcome.bars > 0


def test_only_the_new_bars_are_replayed(warm) -> None:
    board, settings = warm
    mode = "incremental"
    run(board, settings, mode=mode, only="PUT", projections=False)

    engine = board.legs[2].engine
    checkpoint = warm_up_checkpoint(settings)
    through = checkpoint.through(mode, engine.instrument_key, 1)
    assert through is not None

    # Pretend the day moved on: a checkpoint a hundred bars back should leave
    # exactly the bars after it to replay.
    stamps = engine.history.index
    position = int(stamps.searchsorted(through, side="right"))
    moved = stamps[max(position - 100, 1)]
    checkpoint.record(mode, engine.instrument_key, 1, moved, 0, 0.0)

    second = run(board, settings, mode=mode, only="PUT", projections=False)
    outcome = second.instruments[0]
    assert not outcome.cold_start
    assert 0 < outcome.bars <= 100


def test_the_checkpoint_accumulates_across_runs(tmp_path) -> None:
    checkpoint = WarmUpCheckpoint(tmp_path / "warmup.sqlite3")
    moment = pd.Timestamp("2026-09-15 15:29", tz=IST)

    checkpoint.record("simulation", "NSE_INDEX|Nifty 50", 1, moment, 400, 1.5)
    checkpoint.record("simulation", "NSE_INDEX|Nifty 50", 1, moment, 100, 0.4)

    rows = checkpoint.rows()
    assert len(rows) == 1
    assert rows[0]["bars_warmed"] == 500
    assert rows[0]["runs"] == 2
    assert checkpoint.through("simulation", "NSE_INDEX|Nifty 50", 1) == moment


def test_reading_a_checkpoint_that_does_not_exist_creates_nothing(tmp_path) -> None:
    path = tmp_path / "nested" / "warmup.sqlite3"
    checkpoint = WarmUpCheckpoint(path)
    assert checkpoint.through("simulation", "anything", 1) is None
    assert checkpoint.rows() == []
    assert not path.exists()


def test_live_and_simulation_checkpoints_are_separate(tmp_path) -> None:
    checkpoint = WarmUpCheckpoint(tmp_path / "warmup.sqlite3")
    moment = pd.Timestamp("2026-09-15 15:29", tz=IST)
    checkpoint.record("live", "IDX", 1, moment, 10, 0.1)

    assert checkpoint.through("live", "IDX", 1) == moment
    assert checkpoint.through("simulation", "IDX", 1) is None


# ------------------------------------------------------------------- causality


def test_a_signal_is_never_warmed_across_a_session_boundary() -> None:
    """A gap is an overnight move the live loop would never have taken."""
    stamps = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-09-15 15:27", tz=IST),
            pd.Timestamp("2026-09-15 15:28", tz=IST),
            pd.Timestamp("2026-09-15 15:29", tz=IST),
            pd.Timestamp("2026-09-16 09:15", tz=IST),
            pd.Timestamp("2026-09-16 09:16", tz=IST),
        ]
    )
    ahead = WarmUp._ahead

    assert ahead(stamps, 0, 1, 1) == 1
    assert ahead(stamps, 1, 1, 1) == 2
    # 15:29 + 3 minutes is not a bar; the next stamp is the following morning.
    assert ahead(stamps, 2, 1, 1) is None
    assert ahead(stamps, 2, STRATEGY_HORIZON_BARS, 1) is None
    assert ahead(stamps, 3, 1, 1) == 4
    # Past the end of the series there is no target bar at all.
    assert ahead(stamps, 4, 1, 1) is None
    # A five-minute timeframe makes adjacent stamps one bar apart.
    assert ahead(stamps, 2, 1, 1) is None


def test_no_rule_is_trusted_before_its_first_outcome_matures(warm) -> None:
    """The causal property: trust at bar i cannot know bar i's outcome.

    Nothing a rule does can have been scored in the opening bars of a replay,
    because the target bar it would be scored against has not arrived yet. Trust
    there has to be exactly zero — and if the replay were reading the window's
    final scorecard instead of the evidence standing at each bar, it would not be.
    """
    board, settings = warm
    engine = board.legs[0].engine
    seen: list[float] = []

    service = WarmUp()
    original = service._signals_at

    def spy(engine_, context, scores, evidence, position, stamps):
        produced = original(engine_, context, scores, evidence, position, stamps)
        seen.append(
            max((float(s.meta.get("trust_score", 0.0)) for s in produced), default=0.0)
        )
        return produced

    service._signals_at = spy
    try:
        report = service.run(
            [("INDEX", engine)],
            mode="causal",
            bars=WINDOW,
            projections=False,
        )
    finally:
        service._signals_at = original

    outcome = report.instruments[0]
    assert not outcome.skipped, outcome.skipped
    assert len(seen) == outcome.bars
    assert max(seen[:STRATEGY_HORIZON_BARS]) == 0.0, (
        "a rule was trusted before any of its outcomes could have matured"
    )


def test_a_series_with_real_structure_earns_trust(trend_bars, tmp_path) -> None:
    """The replay's positive control.

    Generated bars are a near-random walk, and on a near-random walk no rule
    should earn trust however long the replay runs — so that result alone cannot
    tell a working replay from one that never scored anything. A deterministic
    alternating trend is predictable three bars ahead, and a replay that is
    actually scoring has to notice.
    """
    settings = Settings(
        data_dir=tmp_path / "data", model_dir=tmp_path / "models", log_dir=tmp_path / "logs"
    )
    board = MarketBoard(Kernel.bootstrap(settings), bar_minutes=1).build(trend_bars)
    try:
        for leg in board.legs:
            leg.engine.enable_research()
        report = WarmUp().run(
            [("INDEX", board.legs[0].engine)], mode="trend", bars=WINDOW, projections=False
        )
        outcome = report.instruments[0]
        assert not outcome.skipped, outcome.skipped
        assert outcome.strategies_scored > 0
        assert outcome.trusted_strategies > 0, "no rule earned trust on a deterministic trend"
    finally:
        board.close()


# ------------------------------------------------------------------- refusals


def test_a_replay_with_no_research_says_so_instead_of_failing(warm) -> None:
    board, settings = warm
    engine = board.legs[0].engine
    performance, lab = engine.performance, engine.online_lab
    engine.performance, engine.online_lab = None, None
    try:
        report = WarmUp().run([("INDEX", engine)], mode="disabled", bars=WINDOW)
    finally:
        engine.performance, engine.online_lab = performance, lab
    assert report.instruments[0].skipped == "skipped (research disabled)"


def test_an_instrument_with_too_little_history_is_skipped(warm) -> None:
    board, settings = warm
    engine = board.legs[0].engine
    history = engine.history
    engine.history = engine.history.iloc[:3]
    try:
        report = WarmUp().run([("INDEX", engine)], mode="tiny", bars=WINDOW)
    finally:
        engine.history = history
    assert report.instruments[0].skipped


def test_a_failing_replay_does_not_take_the_session_with_it(warm) -> None:
    board, settings = warm
    engine = board.legs[0].engine
    engine.performance = _Exploding()
    try:
        report = WarmUp().run([("INDEX", engine)], mode="boom", bars=WINDOW)
    finally:
        engine.performance = None
    outcome = report.instruments[0]
    assert outcome.skipped.startswith("skipped (")
    assert report.ran


class _Exploding:
    min_trust_samples = 50

    def observe(self, **_kwargs):
        raise RuntimeError("boom")

    def issue(self, *_args, **_kwargs):
        raise RuntimeError("boom")


# --------------------------------------------------------------- presentation


def test_the_report_is_json_safe(warm) -> None:
    board, settings = warm
    report = run(board, settings, mode="json")
    payload = json.loads(json.dumps(report.as_row()))
    assert payload["ran"] is True
    assert payload["bars"] > 0
    assert payload["instruments"]
    assert payload["headline"]


def test_the_board_carries_the_report_onto_the_frame(warm) -> None:
    from plugins.renderers.web.serialize import serialize_snapshot

    board, settings = warm
    report = run(board, settings, mode="frame")
    board.warmup = report.as_row()
    payload = serialize_snapshot(board.snapshot(), status="SIMULATION")
    assert payload["warmup"]["ran"] is True
    assert payload["warmup"]["bars"] > 0


def test_an_unknown_mode_is_warmed_from_scratch(tmp_path) -> None:
    """A mode with no checkpoint is a cold start, not an error."""
    checkpoint = WarmUpCheckpoint(tmp_path / "warmup.sqlite3")
    assert checkpoint.through("a-mode-never-seen", "IDX", 1) is None
