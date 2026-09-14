"""Contract and causality tests for the advanced mathematical strategy packs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.settings import Settings
from kernel import Kernel
from plugins.features.technical import build_features
from plugins.sources.simulated.series import generate_candles
from plugins.strategies.base import StrategyContext
from plugins.strategies.catalog import StrategyCatalog
from plugins.strategies.regime.rules import DirectionalEntropyStrategy
from plugins.strategies.statistical_anchor.rules import AnchorSpreadReversionStrategy

ADVANCED_PACKS = {"channels", "flow", "regime", "statistical_anchor"}
ADVANCED_NAMES = {
    "keltner_continuation",
    "regression_channel_breakout",
    "channel_pressure",
    "chaikin_flow_trend",
    "money_flow_reversal",
    "obv_divergence",
    "hurst_adaptive",
    "directional_entropy",
    "volatility_state_rotation",
    "anchor_spread_reversion",
    "relative_strength_rotation",
}


@pytest.fixture(scope="module")
def advanced_catalog() -> StrategyCatalog:
    catalog = StrategyCatalog.from_kernel(Kernel.bootstrap(Settings()))
    catalog.packs = [pack for pack in catalog.packs if pack.name in ADVANCED_PACKS]
    return catalog


@pytest.fixture(scope="module")
def market_context() -> StrategyContext:
    bars = generate_candles(days=6, seed=811, with_volume=True)
    anchor = generate_candles(days=6, seed=812, with_volume=True)["close"]
    anchor = anchor / float(anchor.iloc[0]) * float(bars["close"].iloc[0])
    return StrategyContext(
        bars=bars,
        features=build_features(bars),
        extras={"anchor_close": anchor},
    )


def test_advanced_packs_are_discovered_with_unique_capabilities(
    advanced_catalog: StrategyCatalog,
) -> None:
    assert {pack.name for pack in advanced_catalog.packs} == ADVANCED_PACKS
    assert set(advanced_catalog.names()) == ADVANCED_NAMES
    assert len(advanced_catalog.names()) == len(set(advanced_catalog.names()))

    kernel = Kernel.bootstrap(Settings())
    for pack_name in ADVANCED_PACKS:
        entry = kernel.registry.get(f"strategy:{pack_name}")
        advertised = {
            capability.removeprefix("strategy:")
            for capability in entry.manifest.provides
            if capability.startswith("strategy:")
        }
        assert advertised == set(kernel.build(entry.handle).names)


def test_every_advanced_strategy_returns_finite_bounded_conviction(
    advanced_catalog: StrategyCatalog,
    market_context: StrategyContext,
) -> None:
    for strategy in advanced_catalog.all():
        score = strategy.score(market_context)
        assert score.index.equals(market_context.features.index), strategy.name
        assert np.isfinite(score.to_numpy(dtype="float64")).all(), strategy.name
        assert score.abs().max() <= 1.0, strategy.name


def test_advanced_scores_are_causal_under_future_appends(
    advanced_catalog: StrategyCatalog,
    market_context: StrategyContext,
) -> None:
    cutoff = 900
    prefix_bars = market_context.bars.iloc[:cutoff]
    prefix_anchor = market_context.extras["anchor_close"].iloc[:cutoff]
    prefix = StrategyContext(
        bars=prefix_bars,
        features=build_features(prefix_bars),
        extras={"anchor_close": prefix_anchor},
    )

    for strategy in advanced_catalog.all():
        prefix_value = float(strategy.score(prefix).iloc[-1])
        full_value = float(strategy.score(market_context).iloc[cutoff - 1])
        assert prefix_value == pytest.approx(full_value, abs=1e-12), strategy.name


def test_statistical_strategies_are_neutral_without_an_anchor(
    advanced_catalog: StrategyCatalog,
    market_context: StrategyContext,
) -> None:
    unpaired = StrategyContext(bars=market_context.bars, features=market_context.features)
    statistical = [
        strategy for strategy in advanced_catalog.all() if strategy.category == "statistical"
    ]
    assert statistical
    for strategy in statistical:
        assert strategy.score(unpaired).eq(0.0).all()


def test_directional_entropy_tracks_a_predictable_run() -> None:
    index = pd.date_range("2026-01-01", periods=40, freq="min", tz="UTC")
    features = pd.DataFrame(
        {"ret_1": 0.001, "efficiency_ratio_10": 1.0},
        index=index,
    )
    context = StrategyContext(bars=pd.DataFrame(index=index), features=features)
    score = DirectionalEntropyStrategy(window=20).score(context)
    assert score.iloc[-1] > 0.95


def test_anchor_spread_fades_an_idiosyncratic_price_spike() -> None:
    rng = np.random.default_rng(41)
    index = pd.date_range("2026-01-01", periods=260, freq="min", tz="UTC")
    anchor = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.001, len(index)))), index=index)
    asset = anchor * np.exp(rng.normal(0.0, 0.0008, len(index)))
    asset.iloc[-1] *= 1.04
    bars = pd.DataFrame({"close": asset}, index=index)
    context = StrategyContext(
        bars=bars,
        features=pd.DataFrame(index=index),
        extras={"anchor_close": anchor},
    )
    score = AnchorSpreadReversionStrategy(window=60, entry_z=1.2).score(context)
    assert score.iloc[-1] < -0.5
