"""The breakeven gate as a plugin.

Provides the ``advisory`` capability: given a board, a verdict for every leg with
the arithmetic that produced it. The gate is the plugin the platform's whole
argument about costs finally becomes actionable through — and because it is a
plugin, a different policy (a spread selector, a premium-selling model) replaces
it by providing the same capability rather than by rewriting the board.
"""

from __future__ import annotations

from core.types import BoardSnapshot, LegVerdict
from kernel import PluginContext, PluginKind, PluginManifest

from .gate import (
    DEFAULT_COST_RATE,
    FLAT,
    LONG,
    assess_option,
    headline,
    verdict_for_option,
    verdict_for_underlying,
)

MANIFEST = PluginManifest(
    name="breakeven_gate",
    kind=PluginKind.ADVISORY,
    description="Per-leg verdicts: does the projected move clear what the leg costs to hold?",
    provides=("advisory",),
    requires=("projection",),
    tags=("costs", "options", "breakeven"),
    params={"cost_rate": DEFAULT_COST_RATE, "horizon_bars": 3},
)


class BreakevenAdvisor:
    """Turns a board into per-leg verdicts."""

    def __init__(
        self,
        cost_rate: float = DEFAULT_COST_RATE,
        horizon_bars: int = 3,
        hurdle_bps: float = 0.0,
    ) -> None:
        self.cost_rate = float(cost_rate)
        self.horizon_bars = int(max(horizon_bars, 1))
        self.hurdle_bps = float(hurdle_bps)

    def assess(self, board: BoardSnapshot, bar_minutes: int = 1) -> list[LegVerdict]:
        """A verdict for every leg on the board, index included.

        The horizon is the last projected bar, so the requirement and the
        projection are quoted over the same span. Comparing a one-bar requirement
        against a three-bar projection would be the easiest way to talk this
        platform into a trade it has not earned.
        """
        horizon_minutes = self.horizon_bars * max(int(bar_minutes), 1)
        projected = self._projected_move_bps(board)

        verdicts: list[LegVerdict] = []
        for leg in board.legs:
            if leg.is_option:
                verdict = self._assess_option_leg(leg, board.spot, projected, horizon_minutes)
            else:
                verdict = verdict_for_underlying(
                    label=leg.label,
                    projected_move_bps=projected,
                    hurdle_bps=self._hurdle_bps(leg),
                    conviction=leg.snapshot.conviction,
                )
            verdicts.append(verdict)
        return verdicts

    def _hurdle_bps(self, leg) -> float:
        """The round trip for the underlying: configured, or the model's own."""
        if self.hurdle_bps:
            return self.hurdle_bps
        predictions = leg.snapshot.predictions
        return float(predictions[0].hurdle_bps) if predictions else 0.0

    def advise(self, board: BoardSnapshot, bar_minutes: int = 1) -> BoardSnapshot:
        """Attach verdicts and a headline to the board, and return it."""
        verdicts = self.assess(board, bar_minutes=bar_minutes)
        by_label = {verdict.label: verdict for verdict in verdicts}
        for leg in board.legs:
            leg.verdict = by_label.get(leg.label)
        every = [leg.verdict for leg in board.legs if leg.verdict is not None]
        board.headline = headline(every)
        return board

    # ------------------------------------------------------------------ internals

    def _projected_move_bps(self, board: BoardSnapshot) -> float:
        """The underlying's projected move over the horizon, in basis points."""
        spot_leg = board.spot_leg
        if spot_leg is None or not spot_leg.snapshot.projections:
            return 0.0
        path = spot_leg.snapshot.projections
        anchor = path[0].open
        end = path[min(self.horizon_bars, len(path)) - 1].close
        return (end / anchor - 1.0) * 10_000.0 if anchor else 0.0

    def _assess_option_leg(
        self,
        leg,
        spot: float,
        projected: float,
        horizon_minutes: float,
    ) -> LegVerdict:
        """Requirement for one leg, quoted in moves of the **underlying**.

        ``spot`` is the index level, not the leg's premium — and that distinction
        is the whole calculation. A premium divided by its own delta against
        itself gives a requirement in the tens of thousands of basis points, which
        reads as a number and is nonsense: a 79-rupee premium on a 24,000 index
        needs about 68bp of travel, not 21,000.
        """
        greeks = leg.greeks or {}
        premium = float(greeks.get("premium", 0.0))
        delta = float(greeks.get("delta", 0.0))
        theta = float(greeks.get("theta_per_minute", 0.0))
        strike = float(greeks.get("strike", leg.strike))
        if spot <= 0:
            return LegVerdict(
                label=leg.label,
                action="FLAT",
                required_move_bps=float("inf"),
                projected_move_bps=projected,
                reason="no underlying price to measure the requirement against",
            )

        assessment = assess_option(
            premium=premium,
            delta=delta,
            spot=spot,
            theta_per_minute=theta,
            horizon_minutes=horizon_minutes,
            cost_rate=self.cost_rate,
        )
        return verdict_for_option(
            label=leg.label,
            kind=leg.kind,
            assessment=assessment,
            projected_move_bps=projected,
            iv=float(greeks.get("iv", 0.0)),
            realised_vol=float(greeks.get("realised_vol", 0.0)),
            strike=strike,
            spot=spot,
            strike_step=float(greeks.get("strike_step", 50.0)),
        )


def build(
    ctx: PluginContext,
    cost_rate: float = DEFAULT_COST_RATE,
    horizon_bars: int = 3,
    **params,
) -> BreakevenAdvisor:
    return BreakevenAdvisor(
        cost_rate=cost_rate,
        horizon_bars=horizon_bars,
        hurdle_bps=ctx.settings.cost_hurdle_bps(),
    )


__all__ = ["FLAT", "LONG", "MANIFEST", "BreakevenAdvisor", "build"]
