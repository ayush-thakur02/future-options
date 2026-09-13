"""The arithmetic behind a verdict.

One question, asked three ways: **does the move this instrument needs clear the
move it is projected to get, after paying for the wait?**

For the underlying, the cost is the round trip — the platform's original hurdle.
For an option it is more explicit and less forgiving: you pay the premium up
front, the delta converts a move in the underlying into premium, and theta bills
you for every minute you hold. So the requirement becomes

    required_bps = (premium + round-trip cost + theta × horizon)
                   ÷ (|delta| × spot) × 10_000

and the trade is only there when the projected move over the same horizon is
larger. On a scalping horizon it usually is not, by an order of magnitude. That
is the finding, not a failure of the search: the panel exists so the number can
be seen rather than assumed away.

Everything here is a pure function of numbers that came from elsewhere, which is
what makes it testable against a hand-computed example and cheap enough to run on
every refresh.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.types import LegVerdict
from plugins.sources.option_chain.chain import CALL
from plugins.sources.option_chain.pricing import breakeven_move_bps

LONG = "LONG"
SHORT = "SHORT"
FLAT = "FLAT"

# Round-trip cost as a fraction of premium: brokerage both ways, exchange
# charges, STT on the sell leg, GST, plus a spread that is usually the largest
# single item on an option. Deliberately pessimistic — see backtest/costs.py.
DEFAULT_COST_RATE = 0.012
# Implied vol must exceed realised by this much before selling premium is even
# considered. Below it the seller is not being paid for the tail risk taken on.
IV_RICHNESS = 1.15


@dataclass(slots=True)
class OptionAssessment:
    """One option leg's requirement, in the units the decision is made in."""

    required_move_bps: float
    premium_cost: float
    theta_cost_bps: float
    cost_bps: float

    @property
    def is_reachable(self) -> bool:
        return self.required_move_bps < 1_000.0


def assess_option(
    premium: float,
    delta: float,
    spot: float,
    theta_per_minute: float,
    horizon_minutes: float,
    cost_rate: float = DEFAULT_COST_RATE,
) -> OptionAssessment:
    """What the underlying has to do for one long option to pay for itself.

    Theta is folded into the requirement rather than reported beside it. A leg
    that costs 3bp of theta over the horizon and needs 100bp of travel is one
    number, not two, and splitting them invites reading the premium-only figure
    as the hurdle.
    """
    if premium <= 0 or spot <= 0:
        return OptionAssessment(float("inf"), 0.0, 0.0, 0.0)

    cost = premium * max(float(cost_rate), 0.0)
    theta_total = abs(float(theta_per_minute)) * max(float(horizon_minutes), 0.0)
    # A leg cannot lose more than it is worth, so the decay billed over a horizon
    # is capped at the premium itself.
    theta_total = min(theta_total, premium)
    required = breakeven_move_bps(premium + cost + theta_total, abs(delta), spot)

    scale = abs(delta) * spot
    theta_bps = (theta_total / scale) * 10_000.0 if scale else 0.0
    cost_bps = (cost / scale) * 10_000.0 if scale else 0.0
    return OptionAssessment(
        required_move_bps=required,
        premium_cost=premium + cost + theta_total,
        theta_cost_bps=theta_bps,
        cost_bps=cost_bps,
    )


def verdict_for_option(
    label: str,
    kind: str,
    assessment: OptionAssessment,
    projected_move_bps: float,
    iv: float,
    realised_vol: float,
    strike: float,
    spot: float,
    strike_step: float,
) -> LegVerdict:
    """Turn a requirement and a projection into an action, with its reason.

    Three outcomes, and the conservative one is the default:

    * ``LONG`` — the leg is on the side of the projection and the requirement is
      cleared.
    * ``SHORT`` — the leg is on the *opposite* side, implied vol is richer than
      realised, and the strike is at least one step out of the money. Writing it
      collects the premium this projection says will not be paid.
    * ``FLAT`` — everything else, which is almost always.
    """
    up = projected_move_bps >= 0
    on_side = (kind == CALL) == up
    edge = abs(projected_move_bps) - assessment.required_move_bps
    rich = iv > 0 and realised_vol > 0 and iv > realised_vol * IV_RICHNESS
    out_of_money = abs(strike - spot) >= strike_step * 0.5

    if on_side and edge > 0:
        return LegVerdict(
            label=label,
            action=LONG,
            required_move_bps=assessment.required_move_bps,
            projected_move_bps=projected_move_bps,
            theta_cost_bps=assessment.theta_cost_bps,
            cost_bps=assessment.cost_bps,
            iv=iv,
            realised_vol=realised_vol,
            reason=f"needs {assessment.required_move_bps:.0f}bp, projected {abs(projected_move_bps):.0f}bp",
        )

    if not on_side and rich and out_of_money:
        return LegVerdict(
            label=label,
            action=SHORT,
            required_move_bps=assessment.required_move_bps,
            projected_move_bps=projected_move_bps,
            theta_cost_bps=assessment.theta_cost_bps,
            cost_bps=assessment.cost_bps,
            iv=iv,
            realised_vol=realised_vol,
            reason=(
                f"IV {iv:.1%} over realised {realised_vol:.1%}, projection points the other way; "
                f"the underlying must travel {assessment.required_move_bps:.0f}bp against you"
            ),
        )

    if not on_side:
        detail = "projection points the other way"
    elif not assessment.is_reachable:
        detail = "no realistic move pays for this leg"
    else:
        detail = f"needs {assessment.required_move_bps:.0f}bp, projected {abs(projected_move_bps):.0f}bp"
    return LegVerdict(
        label=label,
        action=FLAT,
        required_move_bps=assessment.required_move_bps,
        projected_move_bps=projected_move_bps,
        theta_cost_bps=assessment.theta_cost_bps,
        cost_bps=assessment.cost_bps,
        iv=iv,
        realised_vol=realised_vol,
        reason=detail,
    )


def verdict_for_underlying(
    label: str,
    projected_move_bps: float,
    hurdle_bps: float,
    conviction: float,
) -> LegVerdict:
    """The index or future leg: the original cost hurdle, no delta to help it."""
    edge = abs(projected_move_bps) - hurdle_bps
    if abs(conviction) > 0.05 and edge > 0:
        action = LONG
        reason = f"needs {hurdle_bps:.1f}bp, projected {abs(projected_move_bps):.1f}bp"
    else:
        action = FLAT
        reason = (
            f"needs {hurdle_bps:.1f}bp, projected {abs(projected_move_bps):.1f}bp"
            if abs(conviction) > 0.05
            else "no directional view"
        )
    return LegVerdict(
        label=label,
        action=action,
        required_move_bps=float(hurdle_bps),
        projected_move_bps=float(projected_move_bps),
        reason=reason,
    )


def headline(verdicts: list[LegVerdict]) -> str:
    """One line for the top of the board.

    Longs are ranked by edge and reported first, because that is the only action
    whose edge is measured in the same units as the requirement. A short's edge is
    not a negative version of a long's — it profits from travel that does *not*
    happen — so ranking the two together by ``edge_bps`` would quietly pick the
    short every time.
    """
    longs = [verdict for verdict in verdicts if verdict.action == LONG]
    if longs:
        best = max(longs, key=lambda verdict: verdict.edge_bps)
        return f"{best.label} LONG — edge {best.edge_bps:+.0f}bp ({best.reason})"

    shorts = [verdict for verdict in verdicts if verdict.action == SHORT]
    if shorts:
        best = max(shorts, key=lambda verdict: verdict.iv - verdict.realised_vol)
        return (
            f"{best.label} SHORT — IV {best.iv:.1%} over realised {best.realised_vol:.1%}, "
            f"projection points away"
        )

    return "no leg clears its own breakeven — the correct output here is no trade"


__all__ = [
    "DEFAULT_COST_RATE",
    "FLAT",
    "IV_RICHNESS",
    "LONG",
    "SHORT",
    "OptionAssessment",
    "assess_option",
    "headline",
    "verdict_for_option",
    "verdict_for_underlying",
]
