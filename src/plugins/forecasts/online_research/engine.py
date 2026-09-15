"""Causal online learning and an auditable research prediction ledger."""

from __future__ import annotations

import math
import threading
from collections import deque
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta, tzinfo
from pathlib import Path
from typing import Any

from .algorithms import OnlineClassifier, build_algorithms
from .metrics import MetricAccumulator, conservative_trust, summarise
from .persistence import StateRepository
from .types import (
    AlgorithmPrediction,
    AlgorithmScorecard,
    PredictionRecord,
    ResearchAction,
    ResearchSignal,
)


class OnlineResearchLab:
    """Run prequential forecasts and learn only when their outcomes exist.

    ``issue`` stores what each learner believed at the time. ``observe`` is the
    only method that changes learner parameters, and only for records whose
    target time has passed. This makes the causal boundary explicit and keeps a
    restart from erasing either pending forecasts or historical scorecards.
    """

    def __init__(
        self,
        state_path: Path,
        *,
        algorithms: tuple[str, ...] | Mapping[str, Mapping] | None = None,
        rolling_window: int = 100,
        min_trust_samples: int = 50,
        min_trust_for_action: float = 0.15,
        buy_probability: float = 0.56,
        sell_probability: float = 0.44,
        default_cost_bps: float = 0.0,
    ) -> None:
        if not 0.5 < buy_probability < 1.0:
            raise ValueError("buy_probability must be between 0.5 and 1")
        if not 0.0 < sell_probability < 0.5:
            raise ValueError("sell_probability must be between 0 and 0.5")
        self.state_path = Path(state_path)
        self.rolling_window = max(int(rolling_window), 1)
        self.min_trust_samples = max(int(min_trust_samples), 1)
        self.min_trust_for_action = max(0.0, min(1.0, float(min_trust_for_action)))
        self.buy_probability = float(buy_probability)
        self.sell_probability = float(sell_probability)
        self.default_cost_bps = max(0.0, float(default_cost_bps))
        self.algorithms: dict[str, OnlineClassifier] = build_algorithms(algorithms)
        self._records: list[PredictionRecord] = []
        self._sequence = 0
        # Scorecards are carried, not recomputed. `scorecards()` used to rescan
        # every record the lab had ever written, once per issue — so publishing a
        # forecast got slower the more forecasts had been published. A live
        # session barely notices; a warm-up replay, which asks for the same
        # numbers thousands of times from a ledger growing underneath it, cannot
        # finish at all.
        self._all_time: dict[str, MetricAccumulator] = {}
        self._rolling: dict[str, deque[PredictionRecord]] = {}
        self._dirty: dict[str, PredictionRecord] = {}
        # Forecasts still waiting on a target bar. Kept beside the ledger rather
        # than filtered out of it, because "what is due" is asked once per closed
        # bar and the ledger only ever grows — so the same question would get
        # slower every minute the platform ran.
        self._unresolved: dict[str, PredictionRecord] = {}
        self._expired = 0
        self._repository = StateRepository(self.state_path)
        self._lock = threading.RLock()
        self._restore()

    def issue(
        self,
        features: Mapping[str, Any],
        *,
        timestamp: datetime,
        anchor_price: float,
        horizon_min: int,
        instrument: str = "",
        cost_bps: float | None = None,
        persist: bool = True,
    ) -> ResearchSignal:
        """Persist one prediction per algorithm without fitting any learner.

        ``persist=False`` batches the ledger write into the next :meth:`flush`,
        which is what turns a backfill of thousands of predictions from thousands
        of transactions into a handful. Nothing is lost by returning early: the
        records are held and written by the flush, and a replay always flushes
        before it finishes.
        """
        issued_at = _normalise_time(timestamp)
        horizon = int(horizon_min)
        anchor = float(anchor_price)
        if horizon <= 0:
            raise ValueError("horizon_min must be positive")
        if not math.isfinite(anchor) or anchor <= 0.0:
            raise ValueError("anchor_price must be a positive finite number")
        clean_features = _clean_features(features)
        if not clean_features:
            raise ValueError("features must contain at least one finite numeric value")
        target_at = issued_at + timedelta(minutes=horizon)
        charge = self.default_cost_bps if cost_bps is None else max(0.0, float(cost_bps))

        with self._lock:
            trust_by_algorithm = {card.algorithm: card.trust_score for card in self.scorecards()}
            predictions: list[AlgorithmPrediction] = []
            written: list[PredictionRecord] = []
            for name, algorithm in self.algorithms.items():
                probability = _probability(algorithm.predict_proba(clean_features))
                trust = trust_by_algorithm.get(name, 0.0)
                action = self._action(probability, trust)
                prediction_id = self._next_id(name)
                confidence = abs(probability - 0.5) * 2.0
                record = PredictionRecord(
                    prediction_id=prediction_id,
                    algorithm=name,
                    instrument=str(instrument),
                    issued_at=issued_at,
                    target_at=target_at,
                    horizon_min=horizon,
                    anchor_price=anchor,
                    p_up=probability,
                    action=action,
                    confidence=confidence,
                    trust_at_issue=trust,
                    cost_bps=charge,
                    features=dict(clean_features),
                )
                self._records.append(record)
                self._unresolved[record.prediction_id] = record
                written.append(record)
                predictions.append(
                    AlgorithmPrediction(
                        prediction_id=prediction_id,
                        algorithm=name,
                        p_up=probability,
                        action=action,
                        confidence=confidence,
                        trust_score=trust,
                    )
                )
            self._mark_dirty(written)
            if persist:
                self.flush()

        probability, trust = _consensus(predictions)
        return ResearchSignal(
            issued_at=issued_at,
            target_at=target_at,
            instrument=str(instrument),
            horizon_min=horizon,
            p_up=probability,
            action=self._action(probability, trust),
            confidence=abs(probability - 0.5) * 2.0,
            trust_score=trust,
            predictions=tuple(predictions),
        )

    def observe(
        self,
        *,
        timestamp: datetime,
        price: float,
        instrument: str = "",
        persist: bool = True,
    ) -> list[PredictionRecord]:
        """Score every due forecast and then train it on the realised direction."""
        observed_at = _normalise_time(timestamp)
        actual = float(price)
        if not math.isfinite(actual) or actual <= 0.0:
            raise ValueError("price must be a positive finite number")
        selected_instrument = str(instrument)

        with self._lock:
            due = sorted(
                (
                    record
                    for record in self._unresolved.values()
                    if record.instrument == selected_instrument
                    and record.target_at <= observed_at
                ),
                key=lambda record: (record.target_at, record.issued_at, record.algorithm),
            )
            scored: list[PredictionRecord] = []
            for record in due:
                self._unresolved.pop(record.prediction_id, None)
                if record.target_at < observed_at:
                    record.status = "expired"
                    self._expired += 1
                    self._mark_dirty([record])
                    continue
                realised = (actual / record.anchor_price - 1.0) * 10_000.0
                target = int(realised > 0.0)
                position = {
                    ResearchAction.BUY: 1.0,
                    ResearchAction.SELL: -1.0,
                    ResearchAction.HOLD: 0.0,
                }[record.action]
                gross_pnl = position * realised
                record.matured_at = observed_at
                record.status = "scored"
                record.actual_price = actual
                record.actual_return_bps = realised
                record.label_up = target
                record.hit = (record.p_up >= 0.5) == bool(target)
                record.brier = (record.p_up - target) ** 2
                record.gross_pnl_bps = gross_pnl
                record.net_pnl_bps = (
                    gross_pnl - record.cost_bps if position else 0.0
                )
                algorithm = self.algorithms.get(record.algorithm)
                if algorithm is not None:
                    algorithm.update(record.features, target)
                self._accumulate(record)
                self._mark_dirty([record])
                scored.append(record)
            if due and persist:
                self.flush()
            return [_copy_record(record) for record in scored]

    def flush(self) -> None:
        """Write every record touched since the last flush, in one transaction."""
        with self._lock:
            if not self._dirty:
                return
            pending = list(self._dirty.values())
            self._dirty.clear()
            self._repository.save(
                sequence=self._sequence,
                algorithms={
                    name: algorithm.state_dict() for name, algorithm in self.algorithms.items()
                },
                records=pending,
            )

    def scorecards(self) -> list[AlgorithmScorecard]:
        """Rolling and all-time metrics for every installed learner."""
        with self._lock:
            cards: list[AlgorithmScorecard] = []
            for name in self.algorithms:
                all_time = self._accumulator(name).summary()
                window = self._rolling.get(name)
                rolling = summarise(window) if window else summarise(())
                trust = min(
                    conservative_trust(all_time, self.min_trust_samples),
                    conservative_trust(rolling, self.min_trust_samples),
                )
                cards.append(
                    AlgorithmScorecard(
                        algorithm=name,
                        all_time=all_time,
                        rolling=rolling,
                        trust_score=trust,
                    )
                )
            return cards

    def daily_accuracy(
        self,
        day: date | None = None,
        *,
        timezone: tzinfo = UTC,
    ) -> dict[str, int | float | str]:
        """Aggregate matured learner predictions for one local calendar day.

        A market day must be selected after converting each maturity timestamp
        to the market timezone. Grouping the persisted UTC timestamps directly
        would put the first hours of an Indian session on the wrong date.
        """
        selected_day = day or datetime.now(timezone).date()
        with self._lock:
            records = [
                record
                for record in self._records
                if record.is_scored
                and record.matured_at is not None
                and _in_local_day(record.matured_at, selected_day, timezone)
            ]
        samples = len(records)
        hits = sum(record.hit is True for record in records)
        return {
            "date": selected_day.isoformat(),
            "samples": samples,
            "hits": hits,
            "misses": samples - hits,
            "accuracy": hits / samples if samples else 0.0,
        }

    def records(
        self,
        *,
        algorithm: str | None = None,
        scored: bool | None = None,
        limit: int | None = None,
    ) -> list[PredictionRecord]:
        """Read ledger records in issue order."""
        with self._lock:
            selected = [
                record
                for record in self._records
                if (algorithm is None or record.algorithm == algorithm)
                and (scored is None or record.is_scored is scored)
            ]
            if limit is not None:
                selected = selected[-limit:]
            return [_copy_record(record) for record in selected]

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._unresolved)

    @property
    def expired_count(self) -> int:
        with self._lock:
            return self._expired

    def describe(self) -> str:
        cards = self.scorecards()
        if not any(card.all_time.sample_count for card in cards):
            return f"{len(cards)} online algorithms; no matured predictions yet"
        return " | ".join(
            f"{card.algorithm}: n={card.all_time.sample_count} "
            f"acc={card.all_time.accuracy:.1%} trust={card.trust_score:.0%} "
            f"net={card.all_time.net_pnl_bps:+.1f}bp"
            for card in cards
        )

    def _action(self, probability: float, trust: float) -> ResearchAction:
        if trust < self.min_trust_for_action:
            return ResearchAction.HOLD
        if probability >= self.buy_probability:
            return ResearchAction.BUY
        if probability <= self.sell_probability:
            return ResearchAction.SELL
        return ResearchAction.HOLD

    def _next_id(self, algorithm: str) -> str:
        self._sequence += 1
        return f"online-{self._sequence:012d}-{algorithm}"

    def _accumulator(self, algorithm: str) -> MetricAccumulator:
        accumulator = self._all_time.get(algorithm)
        if accumulator is None:
            accumulator = MetricAccumulator()
            self._all_time[algorithm] = accumulator
        return accumulator

    def _accumulate(self, record: PredictionRecord) -> None:
        """Fold one newly scored record into the carried scorecards."""
        self._accumulator(record.algorithm).add(record)
        window = self._rolling.get(record.algorithm)
        if window is None:
            window = deque(maxlen=self.rolling_window)
            self._rolling[record.algorithm] = window
        window.append(record)

    def _mark_dirty(self, records: list[PredictionRecord]) -> None:
        for record in records:
            self._dirty[record.prediction_id] = record

    def _restore(self) -> None:
        payload = self._repository.load()
        if payload is None:
            return
        self._sequence = int(payload.get("sequence", 0))
        self._records = [
            PredictionRecord.from_dict(dict(record)) for record in payload.get("records", [])
        ]
        saved_algorithms = dict(payload.get("algorithms", {}))
        for name, algorithm in self.algorithms.items():
            state = saved_algorithms.get(name)
            if state is not None:
                algorithm.load_state_dict(dict(state))
        # The carried scorecards start empty and are rebuilt from the restored
        # ledger, so a restart reports the same numbers it reported before.
        for record in self._records:
            if record.is_scored:
                self._accumulate(record)
            elif record.status == "expired":
                self._expired += 1
            else:
                self._unresolved[record.prediction_id] = record


def _clean_features(features: Mapping[str, Any]) -> dict[str, float]:
    cleaned: dict[str, float] = {}
    for name, raw in features.items():
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            cleaned[str(name)] = value
    return dict(sorted(cleaned.items()))


def _normalise_time(timestamp: datetime) -> datetime:
    if not isinstance(timestamp, datetime):
        raise TypeError("timestamp must be a datetime")
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def _in_local_day(timestamp: datetime, day: date, timezone: tzinfo) -> bool:
    normalised = timestamp if timestamp.tzinfo is not None else timestamp.replace(tzinfo=UTC)
    return normalised.astimezone(timezone).date() == day


def _probability(value: float) -> float:
    if not math.isfinite(value):
        return 0.5
    return max(1e-6, min(1.0 - 1e-6, float(value)))


def _consensus(predictions: list[AlgorithmPrediction]) -> tuple[float, float]:
    if not predictions:
        return 0.5, 0.0
    weights = [max(item.trust_score, 0.05) for item in predictions]
    probability = sum(item.p_up * weight for item, weight in zip(predictions, weights, strict=True))
    probability /= sum(weights)
    trust = sum(item.trust_score for item in predictions) / len(predictions)
    return probability, trust


def _copy_record(record: PredictionRecord) -> PredictionRecord:
    return replace(record, features=dict(record.features))


__all__ = ["OnlineResearchLab"]
