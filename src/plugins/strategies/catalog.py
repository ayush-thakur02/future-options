"""The strategy catalog: every strategy the loaded packs offer.

This is what the old hard-coded registry became. Instead of a module importing
eighteen classes by hand, the catalog asks the kernel which strategy packs are
registered and lets each describe itself. The practical consequences:

* Adding a strategy is adding a class to a pack and a name to its ``STRATEGIES``
  tuple. Nothing central changes.
* A pack whose requirements are unmet — the ML strategy without a ``forecast``
  capability — is skipped with a recorded reason rather than crashing the build.
* The strategy audit and the live engine see exactly the same set, because both
  go through here.

The catalog is not a plugin. It is shared machinery *of* the strategy layer, in
the same way ``kernel.registry`` is shared machinery of the kernel.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from kernel import Kernel, PluginKind

from .base import CompositeStrategy, Strategy, StrategyPack

# Prior weights for the default ensemble. These encode a view about what works on
# intraday index data — trend and flow carry more weight than oscillator fades —
# and are meant to be replaced by measured weights after a backtest run.
DEFAULT_WEIGHTS: dict[str, float] = {
    "ema_trend": 1.0,
    "supertrend": 0.9,
    "efficiency_trend": 0.8,
    "donchian_breakout": 0.7,
    "orb": 0.6,
    "macd_momentum": 0.8,
    "roc_momentum": 0.7,
    "squeeze_release": 0.7,
    "vol_breakout": 0.6,
    "vol_regime": 0.5,
    "rsi_reversion": 0.7,
    "bollinger_reversion": 0.7,
    "vwap_reversion": 0.6,
    "zscore_reversion": 0.5,
    "range_fade": 0.4,
    "stochastic": 0.4,
    "activity_spike": 0.5,
    "order_flow": 0.9,
    "vol_reversion": 0.4,
}


@dataclass
class StrategyCatalog:
    """Lookup over every strategy the loaded packs offer."""

    packs: list[StrategyPack] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------- population

    @classmethod
    def from_kernel(cls, kernel: Kernel) -> StrategyCatalog:
        """Build every strategy pack the kernel knows about.

        A pack that cannot be built is recorded rather than raised. On a scalping
        platform the ML strategy legitimately cannot be built until models have
        been trained, and a build must still produce a working rule-based engine.
        """
        catalog = cls()
        for entry in kernel.entries(PluginKind.STRATEGY):
            try:
                built = kernel.build(entry.handle)
            except Exception as exc:  # noqa: BLE001 — a pack is optional by definition
                catalog.skipped[entry.handle] = str(exc)
                continue
            if isinstance(built, StrategyPack):
                catalog.add(built)
            else:
                catalog.skipped[entry.handle] = (
                    f"build() returned {type(built).__name__}, expected StrategyPack"
                )
        return catalog

    def add(self, pack: StrategyPack) -> StrategyCatalog:
        self.packs.append(pack)
        return self

    # ----------------------------------------------------------------- access

    def all(self) -> list[Strategy]:
        return [strategy for pack in self.packs for strategy in pack.instances]

    def names(self) -> list[str]:
        return sorted(strategy.name for strategy in self.all())

    def get(self, name: str) -> Strategy:
        """One strategy by name, or the ensemble."""
        if name == "ensemble":
            return self.ensemble()
        for strategy in self.all():
            if strategy.name == name:
                return strategy
        raise KeyError(f"unknown strategy {name!r}; available: {', '.join(self.names())}")

    def pack_of(self, name: str) -> StrategyPack | None:
        for pack in self.packs:
            if name in pack.names:
                return pack
        return None

    def categories(self) -> list[str]:
        return sorted({pack.category for pack in self.packs})

    # ---------------------------------------------------------------- ensemble

    def ensemble(self, weights: dict[str, float] | None = None) -> CompositeStrategy:
        """Blend the available strategies with the given (or default) weights.

        Weights naming a strategy that is not in this build are dropped silently,
        which is what makes the default weight map usable offline where the ML
        pack has no trained artifacts.
        """
        available = {strategy.name: strategy for strategy in self.all()}
        if weights is None:
            configured = {
                name: weight
                for pack in self.packs
                for name, weight in pack.weights.items()
            }
            weights = {
                name: configured.get(
                    name, DEFAULT_WEIGHTS.get(name, strategy.default_weight)
                )
                for name, strategy in available.items()
            }
        composite = CompositeStrategy()
        for name, weight in weights.items():
            strategy = available.get(name)
            if strategy is not None and weight:
                composite.add(strategy, weight)
        return composite

    def require(self, names: Iterable[str]) -> list[Strategy]:
        return [self.get(name) for name in names]

    # ------------------------------------------------------------ diagnostics

    def describe(self) -> list[dict]:
        return [
            {
                "pack": pack.name,
                "category": pack.category,
                "description": pack.description,
                "strategies": list(pack.names),
            }
            for pack in self.packs
        ]

    def summary(self) -> str:
        counts = ", ".join(f"{pack.name} {len(pack)}" for pack in self.packs)
        return f"{len(self.all())} strategies ({counts})"

    def __len__(self) -> int:
        return len(self.all())

    def __contains__(self, name: object) -> bool:
        return str(name) in self.names()

    def __repr__(self) -> str:
        return f"<StrategyCatalog {len(self.packs)} packs, {len(self.all())} strategies>"


__all__ = ["DEFAULT_WEIGHTS", "StrategyCatalog", "StrategyPack"]
