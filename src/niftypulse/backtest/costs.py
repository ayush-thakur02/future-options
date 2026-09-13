"""Transaction cost model for Indian index derivatives.

Costs are the difference between a strategy that looks profitable in research and
one that makes money. NIFTY 50 is an index — it cannot be traded directly — so
the realistic instrument is a futures contract or an option. This models the
futures case, which is the cleaner one.

Rates change with each Union Budget and are set per exchange, so every component
is a parameter rather than a constant. Defaults reflect the current structure:
flat per-order brokerage, STT on the sell leg, exchange transaction charges, GST
on brokerage and exchange fees, SEBI turnover fee, and stamp duty on the buy leg.

Slippage is modelled separately and is usually the largest real cost for a
strategy trading every few minutes. It is expressed per side, so a round trip
pays it twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class CostModel:
    """Round-trip and per-leg cost calculation."""

    # Flat brokerage per executed order, in rupees. Zero for many discount brokers
    # on delivery; roughly 20 for intraday.
    brokerage_per_order: float = 20.0

    # Securities Transaction Tax, charged on the sell leg of futures.
    stt_sell_bps: float = 2.0

    # NSE transaction charge for index futures.
    exchange_txn_bps: float = 0.19

    # SEBI turnover fee.
    sebi_bps: float = 0.01

    # Stamp duty, buy leg only.
    stamp_duty_buy_bps: float = 0.2

    # GST applied to brokerage plus exchange transaction charges.
    gst_rate: float = 0.18

    # Adverse price movement between decision and fill, per side.
    slippage_bps: float = 0.5

    # NIFTY futures lot size, used to convert flat brokerage into a rate.
    lot_size: int = 75

    def per_leg_bps(self, notional: float, side: str) -> float:
        """Cost of one leg in basis points of notional."""
        if notional <= 0:
            return 0.0

        brokerage = (self.brokerage_per_order / notional) * 10_000
        exchange = self.exchange_txn_bps
        gst = (brokerage + exchange) * self.gst_rate

        variable = self.sebi_bps + exchange + gst
        if side == "sell":
            variable += self.stt_sell_bps
        else:
            variable += self.stamp_duty_buy_bps

        return brokerage + variable + self.slippage_bps

    def round_trip_bps(self, notional: float) -> float:
        return self.per_leg_bps(notional, "buy") + self.per_leg_bps(notional, "sell")

    def round_trip_cost(self, notional: float) -> float:
        """Total round-trip cost in rupees for a given notional."""
        return notional * self.round_trip_bps(notional) / 10_000.0

    def breakeven_move_bps(self, notional: float) -> float:
        """Move required just to cover costs — the strategy's true hurdle."""
        return self.round_trip_bps(notional)

    def as_dict(self) -> dict:
        return {
            "brokerage_per_order": self.brokerage_per_order,
            "round_trip_bps_1L": round(self.round_trip_bps(1_000_00), 3),
            "round_trip_bps_10L": round(self.round_trip_bps(10_000_00), 3),
            "slippage_bps_per_side": self.slippage_bps,
            "lot_size": self.lot_size,
        }


@dataclass
class CostSchedule:
    """Time-varying costs, for modelling past regimes in longer backtests."""

    periods: list[tuple[date, CostModel]] = field(default_factory=list)

    def for_date(self, day: date) -> CostModel:
        applicable = [model for start, model in self.periods if day >= start]
        if not applicable:
            return CostModel()
        return applicable[-1]


def default_model(slippage_bps: float = 0.5) -> CostModel:
    return CostModel(slippage_bps=slippage_bps)


def net_expectancy(
    gross_return_bps: float,
    notional: float,
    cost_model: CostModel,
) -> float:
    """Subtract round-trip costs from a gross return expressed in bps."""
    return gross_return_bps - cost_model.round_trip_bps(notional)
