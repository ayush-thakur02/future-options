"""Prequential quality, calibration, and hypothetical outcome metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence

from .types import MetricSummary, PredictionRecord, ResearchAction


def summarise(records: Sequence[PredictionRecord], calibration_bins: int = 10) -> MetricSummary:
    scored = [record for record in records if record.is_scored]
    count = len(scored)
    if not count:
        return MetricSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    labels = [int(record.label_up or 0) for record in scored]
    probabilities = [record.p_up for record in scored]
    hits = sum(bool(record.hit) for record in scored)
    base_rate = sum(labels) / count
    accuracy = hits / count
    majority_rate = max(base_rate, 1.0 - base_rate)
    brier = sum(float(record.brier or 0.0) for record in scored) / count
    calibration_error = expected_calibration_error(probabilities, labels, bins=calibration_bins)

    trades = [record for record in scored if record.action != ResearchAction.HOLD]
    pnl = [float(record.net_pnl_bps or 0.0) for record in trades]
    gross = sum(float(record.gross_pnl_bps or 0.0) for record in trades)
    costs = sum(record.cost_bps for record in trades)
    wins = sum(value > 0.0 for value in pnl)
    losses = sum(value < 0.0 for value in pnl)
    profit = sum(value for value in pnl if value > 0.0)
    loss = -sum(value for value in pnl if value < 0.0)

    return MetricSummary(
        sample_count=count,
        base_rate=base_rate,
        accuracy=accuracy,
        accuracy_lift=accuracy - majority_rate,
        brier_score=brier,
        calibration_error=calibration_error,
        trade_count=len(trades),
        wins=wins,
        losses=losses,
        win_rate=wins / len(trades) if trades else 0.0,
        gross_pnl_bps=gross,
        cost_paid_bps=costs,
        profit_bps=profit,
        loss_bps=loss,
        net_pnl_bps=sum(pnl),
        mean_net_pnl_bps=sum(pnl) / len(trades) if trades else 0.0,
        max_drawdown_bps=max_drawdown(pnl),
    )


def expected_calibration_error(
    probabilities: Sequence[float], labels: Sequence[int], bins: int = 10
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


def max_drawdown(outcomes: Sequence[float]) -> float:
    equity = 0.0
    peak = 0.0
    deepest = 0.0
    for outcome in outcomes:
        equity += outcome
        peak = max(peak, equity)
        deepest = max(deepest, peak - equity)
    return deepest


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


def wilson_lower(rate: float, count: int, z: float = 1.96) -> float:
    if count <= 0:
        return 0.0
    denominator = 1.0 + z * z / count
    centre = rate + z * z / (2.0 * count)
    spread = z * math.sqrt(rate * (1.0 - rate) / count + z * z / (4.0 * count * count))
    return max(0.0, (centre - spread) / denominator)


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
    "conservative_trust",
    "expected_calibration_error",
    "max_drawdown",
    "summarise",
    "wilson_lower",
]
