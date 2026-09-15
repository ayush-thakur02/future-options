"""Pre-open warm-up: replay recent bars so the platform is not starting cold.

Every session used to open with the same two facts: nothing had been scored yet,
and no trust had been earned. The strategies therefore ran at their plain prior
weights and the online learners returned HOLD for everything, because trust gates
action and trust needs matured outcomes that do not exist until the market has
been open for an hour. The first hour of every day went on learning what the
previous session already knew.

This replays the recent past through the same two ledgers the live loop writes to,
in the order the live loop writes them:

* strategies issue on the bars they would have fired on and are scored at their
  exact target bars, so each rule opens the day with a measured hit rate, an
  expected post-cost result and a trust score;
* the online learners issue a forecast every bar for three horizons and are
  trained as each target closes, so their weights and their calibration are
  already earned;
* the projections are recorded and scored, so the forward scorecard has a history
  behind it rather than a blank.

**The trust fed back into the learners is computed as of each bar**, from the
outcomes that had matured by then. Using the final scorecard for every bar would
be both unrealistic and a leak: a trust value that summarises the whole window
encodes the outcome of the very bar it is describing, and a learner handed it can
learn to read the answer.

A checkpoint records how far each instrument has been warmed, so the second start
replays only what is new. That is what makes this get *better* over time rather
than merely get ready — the state persists, and every session adds to it.

The checkpoint is the whole of the resume logic. Deleting ``warmup.sqlite3``
replays the window again, which the strategy ledger absorbs harmlessly — its
records are keyed by instrument, rule, issue time and target, so the same signal
is never counted twice — but the learners' ledger is append-only, so a lost
checkpoint costs a redundant training pass rather than a wrong one.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from core.scoring import conservative_strategy_trust
from plugins.strategies import StrategyContext

from .engine import ACTIVE_THRESHOLD

# Bars the projection is rebuilt from at each step. `project_candles` reads only a
# 14-bar ATR and a 50-bar volatility median, so a slice this long produces exactly
# the candles the whole frame would, without rebuilding a frame per bar.
PROJECTION_LOOKBACK = 60
# AI horizons, in bars, matching what the live loop issues.
AI_HORIZONS = (1, 2, 3)
# Strategy horizon, in bars, matching `Engine._issue_research`.
STRATEGY_HORIZON_BARS = 3
# How often a replay commits. One transaction per bar would be one transaction
# per bar; waiting until the end would risk the whole replay on one failure.
FLUSH_EVERY = 128

Progress = Callable[[str], None]


@dataclass
class InstrumentWarmUp:
    """What one instrument's replay managed."""

    label: str
    instrument: str
    bars: int = 0
    cold_start: bool = False
    from_bar: str = ""
    through_bar: str = ""
    strategies_issued: int = 0
    strategies_scored: int = 0
    ai_issued: int = 0
    ai_scored: int = 0
    projections_scored: int = 0
    trusted_strategies: int = 0
    seconds: float = 0.0
    skipped: str = ""

    def as_row(self) -> dict:
        return {
            "label": self.label,
            "instrument": self.instrument,
            "bars": self.bars,
            "cold_start": self.cold_start,
            "from_bar": self.from_bar,
            "through_bar": self.through_bar,
            "strategies_issued": self.strategies_issued,
            "strategies_scored": self.strategies_scored,
            "ai_issued": self.ai_issued,
            "ai_scored": self.ai_scored,
            "projections_scored": self.projections_scored,
            "trusted_strategies": self.trusted_strategies,
            "seconds": round(self.seconds, 3),
            "skipped": self.skipped,
        }

    def line(self) -> str:
        if self.skipped:
            return f"{self.label}: {self.skipped}"
        return (
            f"{self.label}: {self.bars:,} bars · {self.strategies_issued:,} signals "
            f"({self.strategies_scored:,} scored) · {self.ai_issued:,} AI forecasts · "
            f"{self.trusted_strategies} rules trusted · {self.seconds:.1f}s"
        )


@dataclass
class WarmUpReport:
    """The whole replay, for the status line and for the dashboard."""

    mode: str = "simulation"
    instruments: list[InstrumentWarmUp] = field(default_factory=list)
    seconds: float = 0.0
    ran: bool = False

    @property
    def bars(self) -> int:
        return sum(item.bars for item in self.instruments)

    @property
    def cold(self) -> bool:
        return any(item.cold_start for item in self.instruments)

    @property
    def trusted_strategies(self) -> int:
        return max((item.trusted_strategies for item in self.instruments), default=0)

    def headline(self) -> str:
        if not self.ran:
            return "not warmed"
        if not self.bars:
            return "already current — nothing new to warm"
        origin = "cold start" if self.cold else "resumed"
        return (
            f"{self.bars:,} bars replayed ({origin}) · "
            f"{self.trusted_strategies} strategies trusted · {self.seconds:.1f}s"
        )

    def as_row(self) -> dict:
        return {
            "headline": self.headline(),
            "mode": self.mode,
            "ran": self.ran,
            "cold": self.cold,
            "bars": self.bars,
            "seconds": round(self.seconds, 3),
            "instruments": [item.as_row() for item in self.instruments],
        }


class _Evidence:
    """Running strategy outcomes, in the shape the engine reads.

    Carried rather than queried. The ledger can score a rule at the end of the
    replay, but the replay needs the number *as it stood at each bar*, and asking
    a database for it once per bar per rule is the difference between a warm-up
    and a coffee break.
    """

    __slots__ = ("accuracy", "hits", "min_trust_samples", "net_pnl_sum", "scored", "trust_score")

    def __init__(self, min_trust_samples: int) -> None:
        self.min_trust_samples = max(int(min_trust_samples), 1)
        self.scored = 0
        self.hits = 0
        self.net_pnl_sum = 0.0
        self.accuracy = 0.0
        self.trust_score = 0.0

    def observe(self, hit: bool, net_pnl_bps: float) -> None:
        self.scored += 1
        self.hits += int(bool(hit))
        self.net_pnl_sum += float(net_pnl_bps)
        self.accuracy = self.hits / self.scored
        self.trust_score = conservative_strategy_trust(
            hits=self.hits,
            samples=self.scored,
            net_pnl_sum=self.net_pnl_sum,
            min_trust_samples=self.min_trust_samples,
        )

    def as_measurement(self) -> dict:
        return {
            "scored": self.scored,
            "hits": self.hits,
            "accuracy": self.accuracy,
            "net_pnl_bps": self.net_pnl_sum,
            "trust_score": self.trust_score,
        }


class WarmUpCheckpoint:
    """How far each instrument has been warmed. Survives restarts by design."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()

    def through(self, mode: str, instrument: str, bar_minutes: int) -> pd.Timestamp | None:
        if not self.path.exists():
            return None
        try:
            with self._lock, self._connect() as database:
                row = database.execute(
                    "SELECT through_ts FROM warmup_progress "
                    "WHERE mode=? AND instrument=? AND bar_minutes=?",
                    (mode, instrument, int(bar_minutes)),
                ).fetchone()
        except sqlite3.Error:
            return None
        if row is None or not row[0]:
            return None
        try:
            return pd.Timestamp(row[0])
        except (TypeError, ValueError):
            return None

    def record(
        self,
        mode: str,
        instrument: str,
        bar_minutes: int,
        through,
        bars: int,
        seconds: float,
    ) -> None:
        if through is None:
            return
        with self._lock, self._connect() as database:
            database.execute(
                """INSERT INTO warmup_progress
                (mode,instrument,bar_minutes,through_ts,bars_warmed,runs,seconds,updated_at)
                VALUES (?,?,?,?,?,1,?,?)
                ON CONFLICT(mode,instrument,bar_minutes) DO UPDATE SET
                    through_ts=excluded.through_ts,
                    bars_warmed=warmup_progress.bars_warmed+excluded.bars_warmed,
                    runs=warmup_progress.runs+1,
                    seconds=excluded.seconds,
                    updated_at=excluded.updated_at""",
                (
                    mode,
                    instrument,
                    int(bar_minutes),
                    str(through),
                    int(bars),
                    float(seconds),
                    datetime.now(UTC).isoformat(),
                ),
            )
            database.commit()

    def rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            with self._lock, self._connect() as database:
                found = database.execute(
                    "SELECT mode,instrument,bar_minutes,through_ts,bars_warmed,runs,updated_at "
                    "FROM warmup_progress ORDER BY instrument"
                ).fetchall()
        except sqlite3.Error:
            return []
        return [
            {
                "mode": str(row[0]),
                "instrument": str(row[1]),
                "bar_minutes": int(row[2]),
                "through_ts": str(row[3]),
                "bars_warmed": int(row[4]),
                "runs": int(row[5]),
                "updated_at": str(row[6]),
            }
            for row in found
        ]

    def reset(self, mode: str | None = None) -> None:
        """Forget the watermark, for one mode or all of them.

        The ledgers are untouched: this only says "replay the window again". The
        learners still remember which bars they were issued, so a reset costs the
        replay and not a second training pass.
        """
        if not self.path.exists():
            return
        with self._lock, self._connect() as database:
            if mode is None:
                database.execute("DELETE FROM warmup_progress")
            else:
                database.execute("DELETE FROM warmup_progress WHERE mode=?", (str(mode),))
            database.commit()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        database = sqlite3.connect(self.path, timeout=20.0)
        database.execute("PRAGMA journal_mode=WAL")
        database.execute("PRAGMA synchronous=FULL")
        database.execute(
            """CREATE TABLE IF NOT EXISTS warmup_progress (
                mode TEXT NOT NULL,
                instrument TEXT NOT NULL,
                bar_minutes INTEGER NOT NULL,
                through_ts TEXT NOT NULL,
                bars_warmed INTEGER NOT NULL DEFAULT 0,
                runs INTEGER NOT NULL DEFAULT 1,
                seconds REAL NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (mode, instrument, bar_minutes)
            )"""
        )
        return database


class WarmUp:
    """Replays recent bars into the research ledgers before a session starts."""

    def __init__(
        self,
        *,
        checkpoint: WarmUpCheckpoint | None = None,
        horizons: int = 1,
        logger: logging.Logger | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.horizons = max(int(horizons), 1)
        self.log = logger or logging.getLogger("niftypulse.warmup")

    def run(
        self,
        targets: Sequence[tuple[str, object]],
        *,
        mode: str = "simulation",
        bars: int = 0,
        projections: bool = True,
        progress: Progress | None = None,
    ) -> WarmUpReport:
        """Warm every ``(label, engine)`` pair.

        ``bars`` caps how far back a **cold** instrument is replayed. An
        instrument with a checkpoint only replays what has appeared since, which
        is what keeps the second start of the day cheap and the state cumulative.
        """
        report = WarmUpReport(mode=mode, ran=True)
        say = progress or (lambda _message: None)
        started = time.perf_counter()

        for label, engine in targets:
            try:
                outcome = self._warm_one(
                    label, engine, mode=mode, bars=bars, projections=projections
                )
            except Exception as exc:  # noqa: BLE001 — a warm-up must never stop a session
                self.log.warning("warm-up failed for %s: %s", label, exc, exc_info=True)
                outcome = InstrumentWarmUp(
                    label=label,
                    instrument=str(getattr(engine, "instrument_key", "") or label),
                    skipped=f"skipped ({type(exc).__name__})",
                )
            report.instruments.append(outcome)
            say(f"    {outcome.line()}")

        report.seconds = time.perf_counter() - started
        return report

    # --------------------------------------------------------------- internals

    def _warm_one(
        self,
        label: str,
        engine,
        *,
        mode: str,
        bars: int,
        projections: bool,
    ) -> InstrumentWarmUp:
        started = time.perf_counter()
        instrument = str(getattr(engine, "instrument_key", "") or label)
        outcome = InstrumentWarmUp(label=label, instrument=instrument)

        history = engine.history
        features = engine.features
        if history.empty or features.empty:
            outcome.skipped = "skipped (no history)"
            return outcome

        ledger = engine.performance
        lab = engine.online_lab
        if ledger is None and lab is None:
            outcome.skipped = "skipped (research disabled)"
            return outcome

        total = len(history)
        stamps = history.index
        span = max(STRATEGY_HORIZON_BARS, max(AI_HORIZONS))
        last = total - 1 - span
        if last < 1:
            outcome.skipped = "skipped (too little history)"
            return outcome

        through = None
        if self.checkpoint is not None:
            through = self.checkpoint.through(mode, instrument, engine.bar_minutes)
        if through is None:
            window = int(bars) if bars > 0 else engine.signal_window
            window = min(max(window, 2), engine.signal_window)
            start = max(last - window + 1, 1)
            outcome.cold_start = True
        else:
            start = int(stamps.searchsorted(through, side="right"))
        if start > last:
            outcome.skipped = "already current"
            outcome.through_bar = str(through)
            outcome.seconds = time.perf_counter() - started
            return outcome

        # The full frame, not the trimmed live window. A score series is indexed by
        # the bar it describes, and the replay walks absolute positions in
        # `history` — trimming to the last N bars here would leave every walked
        # position pointing past the end of the series, and nothing would issue.
        context = StrategyContext(
            bars=history, features=features, extras=engine.strategy_extras()
        )
        scores = self._score_series(engine, context)
        evidence = {
            name: _Evidence(getattr(ledger, "min_trust_samples", 50)) for name in scores
        }
        weights = {rule.name: weight for rule, weight in engine.strategy.components}
        cost_bps = engine.research_cost_bps()

        if lab is not None:
            self._detach_journal(engine.forecaster, detach=True)
        try:
            for position in range(start, last + 1):
                moment = stamps[position].to_pydatetime()
                price = float(history["close"].iloc[position])

                self._score_matured(engine, ledger, lab, outcome, moment, price, evidence)
                if projections:
                    self._score_projections(engine, stamps, position, price, outcome)

                signals = self._signals_at(engine, context, scores, evidence, position, stamps)
                conviction = self._conviction(signals, weights, evidence)
                self._issue(
                    engine, signals, stamps, position, cost_bps, lab, outcome, evidence
                )
                if projections:
                    self._project(engine, history, stamps, position, price, conviction)
                if position % FLUSH_EVERY == 0 and lab is not None:
                    lab.flush()
        finally:
            if lab is not None:
                lab.flush()
                self._detach_journal(engine.forecaster, detach=False)

        outcome.bars = last - start + 1
        outcome.from_bar = str(stamps[start])
        outcome.through_bar = str(stamps[last])
        outcome.trusted_strategies = sum(1 for item in evidence.values() if item.trust_score > 0.0)
        outcome.seconds = time.perf_counter() - started
        # Read the replay's outcomes back into the engine. Without this the
        # ensemble still weights every rule by its plain prior, and the live loop
        # never sees what the replay just learned.
        engine.refresh_rule_stats()
        if self.checkpoint is not None:
            self.checkpoint.record(
                mode, instrument, engine.bar_minutes, stamps[last], outcome.bars, outcome.seconds
            )
        return outcome

    def _score_series(self, engine, context: StrategyContext) -> dict[str, pd.Series]:
        """One vectorised score series per rule that can speak on this data."""
        availability = engine.data_availability(context)
        scores: dict[str, pd.Series] = {}
        for rule in engine.catalog.all():
            if engine.abstention(rule, context, availability):
                continue
            try:
                scores[rule.name] = rule.score(context).fillna(0.0)
            except Exception as exc:  # noqa: BLE001 — a rule that cannot run on history cannot be warmed
                self.log.debug("warm-up skipped %s: %s", rule.name, exc)
        return scores

    def _score_matured(self, engine, ledger, lab, outcome, moment, price, evidence) -> None:
        """Score whatever came due at this bar, before anything is issued from it.

        The order is the whole causal argument: outcomes first, then the readings
        that are allowed to be informed by them.
        """
        if ledger is not None:
            for result in ledger.observe(
                timestamp=moment, price=price, instrument=engine.instrument_key
            ):
                outcome.strategies_scored += 1
                slot = evidence.get(result.strategy)
                if slot is not None:
                    slot.observe(result.hit, result.net_pnl_bps)
        if lab is not None:
            outcome.ai_scored += len(
                lab.observe(
                    timestamp=moment,
                    price=price,
                    instrument=engine.instrument_key,
                    persist=False,
                )
            )

    @staticmethod
    def _score_projections(engine, stamps, position, price, outcome) -> None:
        forecaster = engine.forecaster
        if forecaster is None:
            return
        outcome.projections_scored += len(forecaster.tracker.observe(stamps[position], price))

    def _signals_at(
        self, engine, context, scores, evidence, position, stamps
    ) -> list:
        moment = stamps[position].to_pydatetime()
        produced = []
        for rule in engine.catalog.all():
            series = scores.get(rule.name)
            if series is None or position >= len(series):
                continue
            value = float(series.iloc[position])
            active = abs(value) >= ACTIVE_THRESHOLD
            produced.append(
                engine.build_signal(
                    rule,
                    ts=moment,
                    value=value,
                    state="ACTIVE" if active else "WAIT",
                    reason=rule.describe(value, context) if active else rule.description,
                    measured=evidence[rule.name].as_measurement(),
                )
            )
        return produced

    @staticmethod
    def _conviction(signals, weights, evidence) -> float:
        """The trust-weighted rule blend at one bar, as the engine computes it live."""
        total = weight_sum = 0.0
        for signal in signals:
            trust = evidence[signal.strategy].trust_score if signal.strategy in evidence else 0.0
            weight = weights.get(signal.strategy, 1.0) * (0.75 + 0.5 * trust)
            total += signal.score * weight
            weight_sum += abs(weight)
        return total / weight_sum if weight_sum else 0.0

    def _issue(
        self,
        engine,
        signals,
        stamps,
        position,
        cost_bps,
        lab,
        outcome,
        evidence,
    ) -> None:
        moment = stamps[position].to_pydatetime()
        anchor = float(engine.history["close"].iloc[position])

        if engine.performance is not None and self._ahead(
            stamps, position, STRATEGY_HORIZON_BARS, engine.bar_minutes
        ) is not None:
            active = [signal for signal in signals if signal.meta.get("state") == "ACTIVE"]
            if active:
                outcome.strategies_issued += len(active)
                engine.performance.issue(
                    active,
                    timestamp=moment,
                    anchor_price=anchor,
                    horizon_min=STRATEGY_HORIZON_BARS * engine.bar_minutes,
                    instrument=engine.instrument_key,
                    cost_bps=cost_bps,
                )

        if lab is None:
            return
        measured = {name: slot.as_measurement() for name, slot in evidence.items()}
        row = engine.learning_features(position, signals, measured)
        if not row:
            return
        horizons = tuple(
            bars_ahead * engine.bar_minutes
            for bars_ahead in AI_HORIZONS[: max(int(self.horizons), 1)]
            if self._ahead(stamps, position, bars_ahead, engine.bar_minutes) is not None
        )
        if not horizons:
            return
        issued = lab.issue_horizons(
            row,
            timestamp=moment,
            anchor_price=anchor,
            horizons=horizons,
            instrument=engine.instrument_key,
            cost_bps=cost_bps,
            persist=False,
        )
        outcome.ai_issued += len(issued)

    def _project(self, engine, history, stamps, position, price, conviction) -> None:
        forecaster = engine.forecaster
        if forecaster is None:
            return
        # Only project where every target bar exists inside the replay. A
        # projection left pending past the end would be pruned as a missing bar
        # when the first live bar closed, and the scorecard would report gaps that
        # never happened.
        if self._ahead(stamps, position, max(AI_HORIZONS), engine.bar_minutes) is None:
            return
        window = history.iloc[max(position - PROJECTION_LOOKBACK, 0) : position + 1]
        if len(window) < 2:
            return
        forecaster.refresh(
            bars=window,
            anchor=price,
            conviction=conviction,
            now=stamps[position].to_pydatetime(),
            bar_minutes=engine.bar_minutes,
            record=True,
            issued_at=stamps[position],
        )

    @staticmethod
    def _ahead(stamps, position: int, count: int, bar_minutes: int) -> int | None:
        """The bar ``count`` bars ahead, or None when it is not exactly that far.

        A gap is a session boundary. Warming a signal across one would manufacture
        the overnight outcome the live loop deliberately never takes, and its score
        would be a statement about a move nobody could have traded.
        """
        target = position + count
        if target >= len(stamps):
            return None
        if (stamps[target] - stamps[position]) != pd.Timedelta(minutes=count * bar_minutes):
            return None
        return target

    @staticmethod
    def _detach_journal(forecaster, *, detach: bool) -> None:
        """Keep the live forecast journal out of the replay.

        The journal is an audit of first-issued live forecasts. A backfilled
        projection is neither live nor first, and writing thousands of them would
        bury the record the journal exists to keep.
        """
        tracker = getattr(forecaster, "tracker", None)
        if tracker is None:
            return
        if detach:
            if tracker.journal is not None:
                tracker._warmup_journal = tracker.journal
                tracker.journal = None
        else:
            saved = getattr(tracker, "_warmup_journal", None)
            if saved is not None:
                tracker.journal = saved
                tracker._warmup_journal = None


def warm_up_checkpoint(settings) -> WarmUpCheckpoint:
    """The checkpoint for a run, beside the other research state."""
    return WarmUpCheckpoint(settings.data_dir / "research" / "warmup.sqlite3")


__all__ = [
    "AI_HORIZONS",
    "InstrumentWarmUp",
    "PROJECTION_LOOKBACK",
    "STRATEGY_HORIZON_BARS",
    "WarmUp",
    "WarmUpCheckpoint",
    "WarmUpReport",
    "warm_up_checkpoint",
]
