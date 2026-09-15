"""Prequential quality, calibration, and hypothetical outcome metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence

from core.scoring import max_drawdown, wilson_lower

from .types import MetricSummary, PredictionRecord, ResearchAction

CALIBRATION_BINS = 10


def summarise(records: Sequence[PredictionRecord], calibration_bins: int = CALIBRATION_BINS) -> MetricSummary:
    """Metrics over a sequence of records, via the accumulator.

    Deliberately a thin wrapper: a live scorecard that streams and a backfill that
    batches must report the same numbers, and the surest way to guarantee that is
    for there to be only one implementation of the arithmetic.
    """
    accumulator = MetricAccumulator(bins=calibration_bins)
    accumulator.extend(records)
    return accumulator.summary()


class MetricAccumulator:
    """Streaming version of :func:`summarise`.

    A scorecard was previously a full scan of every record the learner had ever
    produced, recomputed on each issue — so the cost of publishing a prediction
    grew with the number of predictions already made. That is invisible in a
    session and ruinous in a warm-up replay, which asks for exactly the same
    numbers thousands of times in a row from a ledger that is growing underneath
    it. Every quantity here is maintained as it arrives instead.
    """

    __slots__ = (
        "bins",
        "brier_sum",
        "calibration_counts",
        "calibration_probability_sum",
        "calibration_label_sum",
        "cost_sum",
        "count",
        "deepest",
        "equity",
        "gross_sum",
        "hits",
        "label_sum",
        "loss_sum",
        "losses",
        "net_sum",
        "peak",
        "profit_sum",
        "trade_count",
        "wins",
    )

    def __init__(self, bins: int = CALIBRATION_BINS) -> None:
        self.bins = max(int(bins), 1)
        self.calibration_counts = [0] * self.bins
        self.calibration_probability_sum = [0.0] * self.bins
        self.calibration_label_sum = [0] * self.bins
        self.count = 0
        self.label_sum = 0
        self.hits = 0
        self.brier_sum = 0.0
        self.trade_count = 0
        self.wins = 0
        self.losses = 0
        self.gross_sum = 0.0
        self.cost_sum = 0.0
        self.profit_sum = 0.0
        self.loss_sum = 0.0
        self.net_sum = 0.0
        self.equity = 0.0
        self.peak = 0.0
        self.deepest = 0.0

    def add(self, record: PredictionRecord) -> None:
        if not record.is_scored:
            return
        label = int(record.label_up or 0)
        probability = float(record.p_up)
        self.count += 1
        self.label_sum += label
        self.hits += int(bool(record.hit))
        self.brier_sum += float(record.brier or 0.0)

        bucket = min(int(max(0.0, min(1.0, probability)) * self.bins), self.bins - 1)
        self.calibration_counts[bucket] += 1
        self.calibration_probability_sum[bucket] += probability
        self.calibration_label_sum[bucket] += label

        if record.action == ResearchAction.HOLD:
            return
        net = float(record.net_pnl_bps or 0.0)
        self.trade_count += 1
        self.gross_sum += float(record.gross_pnl_bps or 0.0)
        self.cost_sum += float(record.cost_bps)
        self.net_sum += net
        if net > 0.0:
            self.wins += 1
            self.profit_sum += net
        elif net < 0.0:
            self.losses += 1
            self.loss_sum -= net
        # Drawdown needs the series in order, so it is carried rather than stored.
        self.equity += net
        self.peak = max(self.peak, self.equity)
        self.deepest = max(self.deepest, self.peak - self.equity)

    def extend(self, records: Sequence[PredictionRecord]) -> None:
        for record in records:
            self.add(record)

    def summary(self) -> MetricSummary:
        if not self.count:
            return MetricSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        base_rate = self.label_sum / self.count
        accuracy = self.hits / self.count
        majority_rate = max(base_rate, 1.0 - base_rate)
        return MetricSummary(
            sample_count=self.count,
            base_rate=base_rate,
            accuracy=accuracy,
            accuracy_lift=accuracy - majority_rate,
            brier_score=self.brier_sum / self.count,
            calibration_error=self.calibration_error(),
            trade_count=self.trade_count,
            wins=self.wins,
            losses=self.losses,
            win_rate=self.wins / self.trade_count if self.trade_count else 0.0,
            gross_pnl_bps=self.gross_sum,
            cost_paid_bps=self.cost_sum,
            profit_bps=self.profit_sum,
            loss_bps=self.loss_sum,
            net_pnl_bps=self.net_sum,
            mean_net_pnl_bps=self.net_sum / self.trade_count if self.trade_count else 0.0,
            max_drawdown_bps=self.deepest,
        )

    def calibration_error(self) -> float:
        if not self.count:
            return 0.0
        error = 0.0
        for index in range(self.bins):
            bucket = self.calibration_counts[index]
            if not bucket:
                continue
            mean_probability = self.calibration_probability_sum[index] / bucket
            observed_rate = self.calibration_label_sum[index] / bucket
            error += bucket / self.count * abs(mean_probability - observed_rate)
        return error


def expected_calibration_error(
    probabilities: Sequence[float], labels: Sequence[int], bins: int = CALIBRATION_BINS
) -> float:
    if not probabilities:
        return 0.0
    bins = max(int(bins), 1)
    bucketed: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for probability, label in zip(probabilities, labels, strict=True):
        index = min(int(max(0.0, min(1.0, probability)) * bins), bins - 1)
        bucketed[index].append((probability, label))
    total = len(probabilities)
    error = 0.0
    for bucket in bucketed:
        if not bucket:
            continue
        mean_probability = sum(item[0] for item in bucket) / len(bucket)
        observed_rate = sum(item[1] for item in bucket) / len(bucket)
        error += len(bucket) / total * abs(mean_probability - observed_rate)
    return error


def conservative_trust(summary: MetricSummary, min_samples: int = 50) -> float:
    """Evidence-gated trust with lower-bound skill and profitability.

    A calibrated coin flip receives almost no trust. Accuracy is measured over
    the observed majority-class baseline, and profit contributes only when its
    95% lower confidence bound is positive.
    """
    count = summary.sample_count
    if not count:
        return 0.0
    evidence = min(1.0, count / max(int(min_samples), 1))
    lower_accuracy = wilson_lower(summary.accuracy, count)
    majority = max(summary.base_rate, 1.0 - summary.base_rate)
    accuracy_skill = max(0.0, (lower_accuracy - majority) / max(1.0 - majority, 1e-9))

    brier_reference = summary.base_rate * (1.0 - summary.base_rate)
    brier_skill = (
        max(0.0, min(1.0, 1.0 - summary.brier_score / brier_reference))
        if brier_reference > 1e-6
        else 0.0
    )
    calibration_quality = max(0.0, min(1.0, 1.0 - summary.calibration_error / 0.25))
    profit_quality = _profit_lower_bound_quality(summary)
    raw = (
        0.35 * accuracy_skill
        + 0.35 * brier_skill
        + 0.10 * calibration_quality
        + 0.20 * profit_quality
    )
    return max(0.0, min(1.0, evidence * raw))


def _profit_lower_bound_quality(summary: MetricSummary) -> float:
    if summary.trade_count < 2:
        return 0.0
    # The ledger exposes aggregates rather than individual variance. A strict
    # lower bound using drawdown as the risk scale remains conservative and
    # avoids manufacturing confidence from one lucky trade.
    standard_error_proxy = summary.max_drawdown_bps / math.sqrt(summary.trade_count)
    lower_mean = summary.mean_net_pnl_bps - 1.96 * standard_error_proxy
    return math.tanh(max(0.0, lower_mean) / 10.0)


__all__ = [
    "CALIBRATION_BINS",
    "MetricAccumulator",
    "conservative_trust",
    "expected_calibration_error",
    "max_drawdown",
    "summarise",
    "wilson_lower",
]
