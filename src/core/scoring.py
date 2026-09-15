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
    centre, spread, denominator = _wilson(rate, count, z)
    return max(0.0, (centre - spread) / denominator)


def wilson_upper(rate: float, count: int, z: float = 1.96) -> float:
    """Upper bound of the same interval, for the other side of a coin flip."""
    if count <= 0:
        return 0.0
    centre, spread, denominator = _wilson(rate, count, z)
    return min(1.0, (centre + spread) / denominator)


def _wilson(rate: float, count: int, z: float) -> tuple[float, float, float]:
    denominator = 1.0 + z * z / count
    centre = rate + z * z / (2.0 * count)
    spread = z * math.sqrt(rate * (1.0 - rate) / count + z * z / (4.0 * count * count))
    return centre, spread, denominator


def signed_skill(accuracy: float, samples: int, z: float = 1.96) -> float:
    """How far a hit rate is from a coin flip, and on which side of it.

    The sign is only claimed when the whole interval clears 0.5. A rule at 20%
    over five outcomes is evidence of nothing; the same rate over five hundred is
    evidence that the rule is reliably *wrong* — and a reliable loser is worth
    knowing about, which is why this is not clamped at zero. A rule nobody can
    learn anything from returns nothing; a rule that can be counted on to be
    wrong returns a number just as large as one that can be counted on to be
    right.
    """
    lower = wilson_lower(accuracy, samples, z)
    if lower > 0.5:
        return min(1.0, (lower - 0.5) * 2.0)
    upper = wilson_upper(accuracy, samples, z)
    if upper < 0.5:
        return max(-1.0, (upper - 0.5) * 2.0)
    return 0.0


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
    """Trust in one strategy's signals, from only the outcomes seen so far, in [-1, 1].

    Three gates multiplied together, and each one has to be cleared on its own:

    * **Evidence** — a fraction of ``min_trust_samples``, so a rule cannot be
      believed on a handful of outcomes however good they were.
    * **Skill** — how far the hit rate's confidence interval sits from a coin
      flip, *and on which side*. See :func:`signed_skill`: a rule that is reliably
      wrong scores negative rather than scoring nothing.
    * **Profitability** — the mean net result, post-cost, squashed into (-1, 1).
      A rule that loses money after costs earns a negative contribution however
      accurate it is.

    The range is the point. A zero here means "no evidence", not "bad" — and those
    are different things. A signed score lets a consumer weight a rule *down*
    rather than only switching it off, which is what the ensemble does and what
    the online learners are fed.
    """
    if samples <= 0:
        return 0.0
    accuracy = hits / samples
    mean = net_pnl_sum / samples
    evidence = min(1.0, samples / max(int(min_trust_samples), 1))
    skill = signed_skill(accuracy, samples)
    profitability = math.tanh(mean / 10.0)
    return max(-1.0, min(1.0, evidence * (0.75 * skill + 0.25 * profitability)))


def trust_weight(trust: float) -> float:
    """How heavily to weight a rule given its trust score.

    In [0.25, 1.25]: an unproven rule keeps three quarters of its prior, a
    discredited one a quarter, and a proven one a quarter more. Never negative,
    because a negative weight would invert the signal — and inverting a rule that
    has simply been wrong for a while is a much larger claim than the evidence
    supports.
    """
    return 0.75 + 0.5 * max(-1.0, min(1.0, float(trust)))


__all__ = [
    "conservative_strategy_trust",
    "drawdown_from",
    "max_drawdown",
    "signed_skill",
    "trust_weight",
    "wilson_lower",
    "wilson_upper",
]
