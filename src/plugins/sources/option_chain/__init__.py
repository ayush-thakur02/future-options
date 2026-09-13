"""Index options: pricing, the chain, and the legs built from it.

* :mod:`~plugins.sources.option_chain.pricing` — Black-Scholes, in trading time,
  with the breakeven-move arithmetic the platform's cost argument needs
* :mod:`~plugins.sources.option_chain.chain`   — the chain, greeks, open interest
* :class:`~plugins.sources.option_chain.plugin.OptionChainSource` — the capability

Why options belong in a scalping platform at all: an index has no volume and no
order book, so the flow-based strategies fire on nothing. A leg has both. And an
option makes the platform's central question unusually concrete — the premium is
the cost, the delta is the conversion rate, and the theta is the rent paid while
waiting for the move.
"""

from .chain import LegQuote, OptionChain, chain_iv, leg_bars, next_weekly_expiry, synthetic_chain
from .plugin import ChainLeg, OptionChainSource
from .pricing import breakeven_move_bps, bs_price, years_from_minutes

__all__ = [
    "ChainLeg",
    "LegQuote",
    "OptionChain",
    "OptionChainSource",
    "breakeven_move_bps",
    "bs_price",
    "chain_iv",
    "leg_bars",
    "next_weekly_expiry",
    "synthetic_chain",
    "years_from_minutes",
]
