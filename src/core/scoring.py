"""Conservative scoring for a stream of predicted outcomes.

One place where "has this been right often enough to believe it" is answered.
Two packs ask that question — the strategy ledger scores active rules, the online
learners score their own algorithms — and they ask it differently, because they
have different evidence. What they share is the arithmetic underneath: a lower
confidence bound on a hit rate, and a maximum drawdown over an ordered outcome
series.

Keeping it here rather than in either pack matters for one specific reason. A
warm-up replay has to compute trust **as of each bar it walks**, from the
outcomes that had matured by then, or the value it feeds the learners encodes
later results. That means two call sites for the same formula — the batch
scorecard and the incremental replay — and two call sites is exactly where a
formula drifts. Both call this.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def wilson_lower(rate: float, count: int, z: float = 1.96) -> float:
    """Lower bound of the Wilson score interval for a binomial rate.

    The whole point is that it does not reward a small sample. One hit out of one
    attempt scores 0.02, not 1.0 — which is what stops a rule that got lucky twice
    at 09:20 from carrying the same weight as one with two hundred outcomes.
    """
    if count <= 0:
        return 0.0
    denominator = 1.0 + z * z / count
    centre = rate + z * z / (2.0 * count)
    spread = z * math.sqrt(rate * (1.0 - rate) / count + z * z / (4.0 * count * count))
    return max(0.0, (centre - spread) / denominator)


def max_drawdown(outcomes: Sequence[float]) -> float:
    """Deepest peak-to-trough fall of the running sum, as a positive number."""
    equity = peak = deepest = 0.0
    for outcome in outcomes:
        equity += outcome
        peak = max(peak, equity)
        deepest = max(deepest, peak - equity)
    return deepest


def drawdown_from(deepest: float, equity: float, peak: float, outcome: float) -> tuple[float, float, float]:
    """One step of :func:`max_drawdown`, for a caller that cannot keep the series.

    Returns ``(deepest, equity, peak)``. The replay walks thousands of outcomes
    and cannot afford to store them all just to measure a drawdown at the end.
    """
    equity += outcome
    peak = max(peak, equity)
    return max(deepest, peak - equity), equity, peak


def conservative_strategy_trust(
    *,
    hits: int,
    samples: int,
    net_pnl_sum: float,
    min_trust_samples: int = 50,
) -> float:
    """Trust in one strategy's signals, from only the outcomes seen so far.

    Three gates multiplied together, and each one has to be cleared on its own:

    * **Evidence** — a fraction of ``min_trust_samples``, so a rule cannot be
      believed on a handful of outcomes however good they were.
    * **Skill** — the Wilson lower bound of the hit rate, against a coin flip. A
      rule that is right 52% of the time over a large sample earns a little; one
      that is right 52% over five outcomes earns nothing.
    * **Profitability** — the mean net result, post-cost, squashed into [0, 1).
      Zero or negative means the rule is not worth its own costs, and no amount
      of directional accuracy changes that.
    """
    if samples <= 0:
        return 0.0
    accuracy = hits / samples
    mean = net_pnl_sum / samples
    evidence = min(1.0, samples / max(int(min_trust_samples), 1))
    skill = max(0.0, (wilson_lower(accuracy, samples) - 0.5) * 2.0)
    profitability = math.tanh(max(mean, 0.0) / 10.0)
    return max(0.0, min(1.0, evidence * (0.75 * skill + 0.25 * profitability)))


__all__ = [
    "conservative_strategy_trust",
    "drawdown_from",
    "max_drawdown",
    "wilson_lower",
]
