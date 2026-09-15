"""Every source the platform has, blended into one view per leg, with its edge.

The simulator does not invent a signal. It reads the four things the rest of the
platform already computes — the strategy ensemble, the online AI learners, the
batch model's forecast, and the projected path — and blends them into a single
number in [-1, 1] that says how much the leg's own price is expected to move.
Each contribution is kept alongside the blend, so a reader can see which source
carried a decision instead of taking the total on faith.

The edge arithmetic is the part worth reading carefully, because it differs from
the breakeven gate's and the difference is deliberate.

The gate asks whether the option can recover the premium it was bought for: a
240-point move, on a 120-rupee ATM premium. For a position held to expiry that
is the right question. A scalp does not hold to expiry — it buys at 120 and sells
at 123, and the 120 comes back. So this asks the round-trip question instead:

    expected move of the leg's own price  -  what the round trip costs

For an ATM leg a three-bar projection is worth roughly 300bp of premium against
a round trip near 120bp, so the arithmetic says yes. The gate says no. Both are
correct answers to two different questions, and the simulator shows its own
number rather than borrowing the stricter one.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime

from core.scoring import trust_weight
from core.types import Direction

BUY = "BUY"
SELL = "SELL"
HOLD = "HOLD"
EXIT = "EXIT"

# The source the entry gates treat as the forward view.
PROJECTION_SOURCE = "projection"
# Below this a source is noise rather than an opinion, and counting it as
# agreement would let two nearly-flat inputs outvote one strong one.
SOFT_FLOOR = 0.05


def squash(value: float, scale: float) -> float:
    """A bounded opinion from an unbounded one, so no single source can dominate."""
    if not math.isfinite(float(value)):
        return 0.0
    return float(math.tanh(float(value) / (scale + 1e-12)))


@dataclass(slots=True)
class Contribution:
    """One source's input to the blend, kept for the reader."""

    source: str
    value: float
    weight: float
    detail: str = ""

    @property
    def weighted(self) -> float:
        return self.value * self.weight


@dataclass(slots=True)
class LegState:
    """What the simulator reads from one leg, handed over by the board.

    Deliberately a flat record of numbers and small lists rather than a live
    engine: the policy has no business reaching into a running instrument, and a
    record can be constructed in a test without standing one up.

    The board may hand this over as a mapping instead — see :meth:`from_mapping` —
    so the runtime depends on the *shape* of the input rather than on a class in
    this pack. That is what lets another advisory pack provide ``money_simulator``
    without the board changing a line.
    """

    label: str
    kind: str
    instrument: str
    price: float
    spot: float = 0.0
    index_beta: float = 1.0
    signals: tuple = ()
    ai_signal: object | None = None
    predictions: tuple = ()
    projections: tuple = ()
    # The underlying's projected path, which is *not* this leg's own path when
    # the leg is an option: a premium is a levered function of the index, and its
    # own projection moves by whole percents a bar. Reading the premium's path as
    # though it were the index's would report a five-percent move in the
    # underlying every three minutes.
    index_projections: tuple = ()
    conviction: float = 0.0
    atr: float = 0.0
    micro: float = 0.0
    greeks: dict = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping) -> LegState:
        """Build from the board's plain-dict description of a leg."""
        return cls(
            label=str(payload["label"]),
            kind=str(payload.get("kind", "index")),
            instrument=str(payload.get("instrument", "")),
            price=float(payload.get("price", 0.0) or 0.0),
            spot=float(payload.get("spot", 0.0) or 0.0),
            index_beta=float(payload.get("index_beta", 1.0) or 1.0),
            signals=tuple(payload.get("signals") or ()),
            ai_signal=payload.get("ai_signal"),
            predictions=tuple(payload.get("predictions") or ()),
            projections=tuple(payload.get("projections") or ()),
            index_projections=tuple(payload.get("index_projections") or ()),
            conviction=float(payload.get("conviction", 0.0) or 0.0),
            atr=float(payload.get("atr", 0.0) or 0.0),
            micro=float(payload.get("micro", 0.0) or 0.0),
            greeks=dict(payload.get("greeks") or {}),
        )

    @property
    def is_option(self) -> bool:
        return self.kind in {"CE", "PE"}

    @property
    def delta(self) -> float:
        try:
            return float(self.greeks.get("delta", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0


def as_leg_state(item) -> LegState:
    """Accept either a built state or the board's mapping. Both are supported."""
    return item if isinstance(item, LegState) else LegState.from_mapping(item)


@dataclass(slots=True)
class LegView:
    """The blend, and the rupees it is worth if it is right."""

    leg: str
    kind: str
    view: float
    index_view: float
    expected_move_bps: float
    required_move_bps: float
    contributions: tuple[Contribution, ...] = ()
    horizon_minutes: int = 0
    price: float = 0.0
    units: int = 1
    # The sign of the projected move at each bar of the horizon, in the leg's own
    # price. A path that bends back is not a path to hold, and the per-bar view is
    # the only place a learner's disagreement with its own longer horizon shows up.
    horizon_directions: tuple[int, ...] = ()

    @property
    def side(self) -> Direction:
        return Direction.UP if self.view > 0 else Direction.DOWN if self.view < 0 else Direction.FLAT

    @property
    def notional(self) -> float:
        return max(self.price * self.units, 0.0)

    @property
    def edge_bps(self) -> float:
        return abs(self.expected_move_bps) - self.required_move_bps

    @property
    def edge_rupees(self) -> float:
        return self.edge_bps / 10_000.0 * self.notional

    @property
    def leading_source(self) -> str:
        """The source carrying the most weight in this view."""
        if not self.contributions:
            return ""
        best = max(self.contributions, key=lambda item: abs(item.weighted))
        return best.source if abs(best.weighted) > 0.0 else ""

    @property
    def projection_leads(self) -> bool:
        """Whether the projected path, not something else, is what the view rests on."""
        return self.leading_source == PROJECTION_SOURCE

    @property
    def agreement(self) -> tuple[int, int]:
        """Sources leaning with the view, and sources with any opinion at all.

        Compared in **index** space, not in the leg's own. Every contribution is a
        statement about the underlying, while ``view`` is a statement about the
        leg — and for a put those have opposite signs by construction, so testing
        contributions against the leg's view would report a put that every source
        agrees on as agreeing with none of them.

        A source weighted out of the blend does not get to confirm anything, which
        is what lets momentum be switched off without it still counting as a vote.
        """
        leaning = [
            item
            for item in self.contributions
            if item.weight > 0 and abs(item.value) >= SOFT_FLOOR
        ]
        agree = sum(1 for item in leaning if item.value * self.index_view > 0)
        return agree, len(leaning)

    def as_row(self) -> dict:
        agree, total = self.agreement
        return {
            "leg": self.leg,
            "view": round(self.view, 4),
            "index_view": round(self.index_view, 4),
            "expected_move_bps": round(self.expected_move_bps, 3),
            "required_move_bps": round(self.required_move_bps, 3),
            "edge_bps": round(self.edge_bps, 3),
            "edge_rupees": round(self.edge_rupees, 2),
            "agreement": f"{agree}/{total}" if total else "0/0",
            "leading_source": self.leading_source,
            "projection_leads": self.projection_leads,
            "horizon_directions": list(self.horizon_directions),
            "sources": [
                {
                    "source": item.source,
                    "value": round(item.value, 4),
                    "weight": round(item.weight, 3),
                    "weighted": round(item.weighted, 4),
                    "detail": item.detail,
                }
                for item in self.contributions
            ],
        }


@dataclass(slots=True)
class Decision:
    """What the simulator decided about one leg, and why."""

    leg: str
    action: str
    reason: str
    view: float
    price: float
    at: datetime
    edge_bps: float = 0.0
    units: int = 0
    pnl: float = 0.0

    def as_row(self) -> dict:
        return {
            "leg": self.leg,
            "action": self.action,
            "reason": self.reason,
            "view": round(self.view, 4),
            "price": round(self.price, 2),
            "edge_bps": round(self.edge_bps, 3),
            "units": self.units,
            "pnl": round(self.pnl, 2),
            "at": self.at.isoformat(),
        }


@dataclass
class PolicyWeights:
    """How much each source counts. Renormalised over the sources present."""

    strategy: float = 1.0
    ai: float = 0.8
    model: float = 0.5
    projection: float = 1.5
    # Zero on purpose, and it is the whole of "a tick is not a thesis". Momentum
    # is the last few seconds of tape, so any weight at all lets one favourable
    # print carry a view over the entry threshold — which is a position opened on
    # the tick that just happened rather than on where the next few minutes are
    # projected to go. Raise it to study tick-reactive entries; leave it at zero
    # to trade the forward path.
    momentum: float = 0.0

    def __post_init__(self) -> None:
        if min(self.strategy, self.ai, self.model, self.projection, self.momentum) < 0:
            raise ValueError("policy weights cannot be negative")

    def as_row(self) -> dict:
        return {
            "strategy": self.strategy,
            "ai": self.ai,
            "model": self.model,
            "projection": self.projection,
            "momentum": self.momentum,
        }


class DecisionPolicy:
    """Turns leg state into a view, and a view into an action."""

    def __init__(
        self,
        *,
        weights: PolicyWeights | None = None,
        horizon_bars: int = 3,
        entry_view: float = 0.15,
        min_edge_bps: float = 0.0,
        min_confirmations: int = 1,
        option_cost_rate: float = 0.012,
        projection_scale_bps: float = 4.0,
        units: int = 65,
        index_cost_bps: float = 0.0,
        require_projection: bool = True,
    ) -> None:
        self.weights = weights or PolicyWeights()
        self.horizon_bars = max(int(horizon_bars), 1)
        self.entry_view = abs(float(entry_view))
        self.min_edge_bps = float(min_edge_bps)
        self.min_confirmations = max(int(min_confirmations), 0)
        self.option_cost_rate = max(float(option_cost_rate), 0.0)
        # A four-bar move of four basis points is roughly one "fully convinced"
        # projection, which is the scale the projected path is read on.
        self.projection_scale_bps = max(float(projection_scale_bps), 1e-6)
        self.units = max(int(units), 1)
        self.index_cost_bps = max(float(index_cost_bps), 0.0)
        self.require_projection = bool(require_projection)

    # -------------------------------------------------------------- the view

    def view_for(self, state: LegState, cost_model=None) -> LegView:
        """Blend every available source for one leg."""
        contributions: list[Contribution] = []

        strategy, detail = _strategy_view(state.signals)
        contributions.append(Contribution("strategies", strategy, self.weights.strategy, detail))

        ai = _ai_view(state.ai_signal)
        contributions.append(Contribution("online_ai", ai.value, self.weights.ai, ai.detail))

        model = _model_view(state.predictions)
        contributions.append(Contribution("model", model.value, self.weights.model, model.detail))

        projection_bps = self.projected_index_move_bps(state)
        # A leg with no projected path has no forward view to contribute, so it is
        # weighted out rather than entered as a confident zero — which would
        # dilute the sources that do have something to say.
        has_path = bool(state.index_projections or state.projections)
        contributions.append(
            Contribution(
                PROJECTION_SOURCE,
                squash(projection_bps, self.projection_scale_bps),
                self.weights.projection if has_path else 0.0,
                f"{projection_bps:+.2f}bp over {self.horizon_bars} bars"
                if has_path
                else "no projected path",
            )
        )

        contributions.append(
            Contribution("momentum", _clamp(state.micro), self.weights.momentum, "last seconds of tape")
        )

        index_view = _blend(contributions)
        view = index_view * float(state.index_beta or 1.0)
        move_bps = self.expected_move_bps(state, projection_bps)

        return LegView(
            leg=state.label,
            kind=state.kind,
            view=view,
            index_view=index_view,
            expected_move_bps=move_bps,
            required_move_bps=self.required_move_bps(state, cost_model),
            contributions=tuple(contributions),
            horizon_minutes=self.horizon_bars,
            price=float(state.price),
            units=self.units,
            horizon_directions=self.horizon_directions(state),
        )

    def horizon_directions(self, state: LegState) -> tuple[int, ...]:
        """The sign of the projected move at each bar, **in the leg's own price**.

        Read bar by bar rather than at the end of the path, because a path that
        bends back is not a path to hold: the first two minutes pointing up and
        the third pointing down is a trade whose own forecast says it should be
        closed before its target. It is also where a learner's disagreement with
        its own longer horizon shows up, since the projected path blends the
        online regression's per-horizon moves over the rule-based drift.

        The path is the underlying's, so it is flipped for a put by the same sign
        that flips the view — otherwise a put whose forecast is exactly right
        would read as a forecast pointing the other way.
        """
        path = list(state.index_projections or state.projections)
        if not path:
            return ()
        anchor = float(getattr(path[0], "open", 0.0) or 0.0)
        if anchor <= 0:
            return ()
        beta = float(state.index_beta or 1.0)
        directions: list[int] = []
        for candle in path[: self.horizon_bars]:
            close = float(getattr(candle, "close", 0.0) or 0.0)
            if close <= 0:
                directions.append(0)
                continue
            directions.append(Direction.from_value((close - anchor) * beta).sign)
        return tuple(directions)

    def projected_index_move_bps(self, state: LegState) -> float:
        """The projected move of the underlying over the horizon, in basis points.

        Reads the *underlying's* path rather than the leg's own, which for an
        option are different series entirely.
        """
        path = list(state.index_projections or state.projections)
        if not path:
            return 0.0
        window = path[: self.horizon_bars]
        anchor = float(getattr(window[0], "open", 0.0) or 0.0)
        end = float(getattr(window[-1], "close", 0.0) or 0.0)
        if anchor <= 0 or end <= 0:
            return 0.0
        return (end / anchor - 1.0) * 10_000.0

    def expected_move_bps(self, state: LegState, index_move_bps: float) -> float:
        """The same projection, restated in the leg's own price.

        An option does not move in index basis points. Its premium moves by the
        delta times the underlying's travel, and a 65-unit ATM leg near a
        120-rupee premium turns a three-basis-point index move into several
        hundred basis points of premium. Quoting both legs in index basis points
        would make the option look like a worse index position rather than a
        different instrument.
        """
        if not state.is_option or state.price <= 0:
            return index_move_bps
        delta = state.delta
        if delta == 0.0:
            return 0.0
        spot = float(state.spot or 0.0)
        if spot <= 0:
            return 0.0
        points = spot * index_move_bps / 10_000.0
        return (delta * points) / state.price * 10_000.0

    def required_move_bps(self, state: LegState, cost_model=None) -> float:
        """What the round trip costs, in basis points of the leg's own price.

        An option pays twice for being held: the spread and charges to get in and
        out, and the decay that runs the whole time it is open. Both are on the
        same side of the ledger, so both belong in the same number — reporting
        spread alone would quote a hurdle that a position clears by simply not
        waiting for its own move.
        """
        if state.is_option:
            hurdle = self.option_cost_rate
            if state.price > 0:
                decay = abs(self._theta(state)) * self.horizon_bars
                # A leg cannot decay past zero, so the decay billed over the
                # horizon is capped at the premium it is eating.
                hurdle += min(decay, state.price) / state.price
            return hurdle * 10_000.0
        if cost_model is not None and state.price > 0:
            return float(cost_model.round_trip_bps(state.price * self.units))
        return self.index_cost_bps

    @staticmethod
    def _theta(state: LegState) -> float:
        """Premium lost per trading minute, as a positive cost."""
        try:
            return abs(float(state.greeks.get("theta_per_minute", 0.0) or 0.0))
        except (TypeError, ValueError):
            return 0.0

    # ------------------------------------------------------------ the action

    def action_for(self, view: LegView, has_position: bool) -> tuple[str, str]:
        """Enter, exit, or hold — with the reason, which is the part that matters.

        The gates are ordered so the most specific explanation wins: a reader
        asking "why is it not trading" should get the actual binding constraint
        rather than the first check that happened to fail.

        Two of them exist to answer one complaint: that a position opened on the
        tick that had just gone the right way. The projected path has to lead the
        view, and every bar of that path has to point where the trade does. A
        favourable print can still move the blend — the projection is anchored on
        the live price — but it cannot by itself be the reason for the entry.
        """
        if has_position:
            return HOLD, "holding"

        if abs(view.view) < self.entry_view:
            return HOLD, f"view {view.view:+.2f} below {self.entry_view:.2f}"

        side = view.side
        if side is Direction.FLAT:
            return HOLD, "no directional view"

        if not view.horizon_directions:
            return HOLD, "no projected path to trade against"

        disagreeing = [index for index, sign in enumerate(view.horizon_directions, 1) if sign != side.sign]
        if disagreeing:
            horizon = ", ".join(f"+{index}m" for index in disagreeing)
            return HOLD, f"projected path disagrees at {horizon}"

        if self.require_projection and not view.projection_leads:
            leader = view.leading_source or "nothing"
            return HOLD, f"the projection does not lead this view — {leader} does"

        agree, total = view.agreement
        if agree < self.min_confirmations:
            return HOLD, f"only {agree} source(s) confirm, {self.min_confirmations} required"

        if view.expected_move_bps * side.sign <= 0:
            return HOLD, "projection disagrees with the blended view"

        if view.edge_bps < self.min_edge_bps:
            return HOLD, (
                f"edge {view.edge_bps:+.1f}bp below {self.min_edge_bps:.1f}bp "
                f"(needs {view.required_move_bps:.1f}bp, projects {abs(view.expected_move_bps):.1f}bp)"
            )

        action = BUY if side is Direction.UP else SELL
        return action, (
            f"{side.value} view {view.view:+.2f} on a projected "
            f"{'+'.join(str(sign) for sign in view.horizon_directions)} path, "
            f"edge {view.edge_bps:+.1f}bp ({agree}/{total} sources agree)"
        )


# ----------------------------------------------------------------- helpers


@dataclass(slots=True)
class _Score:
    value: float = 0.0
    detail: str = ""


def _clamp(value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return max(-1.0, min(1.0, number))


def _strategy_view(signals) -> tuple[float, str]:
    """The ensemble's signed score, weighted by each rule's measured trust.

    Trust is signed and may be negative, so a rule that has been reliably wrong
    is weighted *down* rather than merely being switched off. Weighting it
    negatively would invert its signal, which is a much larger claim than the
    evidence supports — a rule can be wrong for a regime and right in the next.
    """
    total = weight_sum = 0.0
    used = active = 0
    for signal in signals or ():
        meta = getattr(signal, "meta", None) or {}
        if meta.get("state") in {"N/A", "ERROR"}:
            continue
        score = _clamp(getattr(signal, "score", 0.0))
        try:
            trust = max(-1.0, min(1.0, float(meta.get("trust_score", 0.0))))
        except (TypeError, ValueError):
            trust = 0.0
        weight = trust_weight(trust)
        total += score * weight
        weight_sum += weight
        used += 1
        if abs(score) >= 0.15:
            active += 1
    if not weight_sum:
        return 0.0, "no strategy reading"
    return _clamp(total / weight_sum), f"{active} active of {used} rules"


def _ai_view(signal) -> _Score:
    """The online learners' ensemble, shrunk by how much they have proved."""
    if signal is None:
        return _Score(0.0, "no online reading")
    p_up = getattr(signal, "p_up", None)
    if p_up is None:
        return _Score(0.0, "no online reading")
    try:
        trust = max(0.0, min(1.0, float(getattr(signal, "trust_score", 0.0))))
        edge = (float(p_up) - 0.5) * 2.0
    except (TypeError, ValueError):
        return _Score(0.0, "no online reading")
    return _Score(_clamp(edge * trust), f"p(up) {float(p_up):.2f}, trust {trust:.2f}")


def _model_view(predictions) -> _Score:
    """The batch ensemble's view, averaged over the horizons it reports."""
    edges: list[float] = []
    for prediction in predictions or ():
        p_up = getattr(prediction, "p_up", None)
        if p_up is None:
            continue
        try:
            edges.append((float(p_up) - 0.5) * 2.0)
        except (TypeError, ValueError):
            continue
    if not edges:
        return _Score(0.0, "no trained model")
    mean = sum(edges) / len(edges)
    return _Score(_clamp(mean), f"{len(edges)} horizon(s), mean edge {mean:+.2f}")


def _blend(contributions: list[Contribution]) -> float:
    """Weighted mean over the sources that actually have something to say.

    Renormalising by the weights *present* rather than by every configured weight
    is what stops a missing model from diluting the blend: offline, with no
    trained artifacts, the four remaining sources should read as a full opinion
    rather than as three quarters of one.
    """
    present = [item for item in contributions if item.weight > 0]
    if not present:
        return 0.0
    total = sum(item.weighted for item in present)
    weight = sum(item.weight for item in present)
    return _clamp(total / weight)


__all__ = [
    "BUY",
    "EXIT",
    "HOLD",
    "PROJECTION_SOURCE",
    "SELL",
    "Contribution",
    "Decision",
    "DecisionPolicy",
    "LegState",
    "LegView",
    "PolicyWeights",
    "SOFT_FLOOR",
    "as_leg_state",
    "squash",
]
