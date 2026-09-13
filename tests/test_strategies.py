"""Tests for the strategy packs and the catalog.

Strategies were the one part of the platform with no tests at all before this,
which is exactly the part where a silent mistake — a rule that returns nothing, a
weight naming a strategy that does not exist — looks like a working dashboard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.settings import Settings
from kernel import Kernel, PluginEntry, PluginKind, PluginManifest
from plugins.features.technical import build_features
from plugins.sources.simulated.series import generate_candles
from plugins.strategies import (
    DEFAULT_WEIGHTS,
    CompositeStrategy,
    StrategyCatalog,
    StrategyContext,
)

ML_PACK = "strategy:ml_forecast"


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    return generate_candles(days=25, seed=31)


@pytest.fixture(scope="module")
def context(bars) -> StrategyContext:
    return StrategyContext(bars=bars, features=build_features(bars))


@pytest.fixture(scope="module")
def catalog() -> StrategyCatalog:
    return StrategyCatalog.from_kernel(Kernel.bootstrap(Settings()))


# ------------------------------------------------------------------ discovery


def test_catalog_finds_every_rule_pack(catalog: StrategyCatalog) -> None:
    assert {pack.name for pack in catalog.packs} == {
        "trend",
        "momentum",
        "reversion",
        "volatility",
    }


def test_catalog_offers_every_rule(catalog: StrategyCatalog) -> None:
    assert len(catalog) == 19
    assert {"ema_trend", "rsi_reversion", "order_flow", "roc_momentum"} <= set(catalog.names())


def test_roc_momentum_is_reachable(catalog: StrategyCatalog) -> None:
    """It was implemented but never registered before this, so its weight never applied."""
    assert "roc_momentum" in catalog
    assert "roc_momentum" in DEFAULT_WEIGHTS


def test_every_advertised_capability_has_a_class(catalog: StrategyCatalog) -> None:
    """A pack cannot advertise a strategy it does not implement, or hide one it does."""
    kernel = Kernel.bootstrap(Settings())
    for entry in kernel.entries(PluginKind.STRATEGY):
        if entry.handle in catalog.skipped:
            continue
        advertised = {
            capability.split(":", 1)[1]
            for capability in entry.manifest.provides
            if capability.startswith("strategy:")
        }
        assert advertised == set(kernel.build(entry.handle).names)


def test_ml_pack_is_skipped_with_a_reason_when_untrained(catalog: StrategyCatalog) -> None:
    """A pack that cannot be built is recorded, not raised: offline builds must run."""
    assert ML_PACK in catalog.skipped
    assert "train" in catalog.skipped[ML_PACK]
    assert "ml" not in catalog.names()


def test_catalog_reports_its_packs(catalog: StrategyCatalog) -> None:
    described = {row["pack"]: row for row in catalog.describe()}
    assert described["trend"]["category"] == "trend"
    assert "ema_trend" in described["trend"]["strategies"]


# ------------------------------------------------------------------- ensemble


def test_ensemble_uses_every_available_weighted_strategy(catalog: StrategyCatalog) -> None:
    ensemble = catalog.ensemble()
    names = {strategy.name for strategy, _ in ensemble.components}
    assert names == set(DEFAULT_WEIGHTS) & set(catalog.names())
    assert len(names) == 19


def test_ensemble_skips_weights_for_absent_strategies(catalog: StrategyCatalog) -> None:
    """The ML strategy is absent offline, so its weight must simply not apply."""
    ensemble = catalog.ensemble({"ml": 1.0, "ema_trend": 1.0})
    assert [strategy.name for strategy, _ in ensemble.components] == ["ema_trend"]


def test_ensemble_of_nothing_is_flat(context: StrategyContext) -> None:
    """An ensemble with no components must return zeros, not raise."""
    scores = CompositeStrategy().score(context)
    assert scores.abs().sum() == 0.0
    assert len(scores) == len(context.features)


def test_get_resolves_the_ensemble(catalog: StrategyCatalog) -> None:
    assert isinstance(catalog.get("ensemble"), CompositeStrategy)


def test_get_unknown_strategy_lists_what_is_available(catalog: StrategyCatalog) -> None:
    with pytest.raises(KeyError, match="available:"):
        catalog.get("nonexistent_rule")


def test_pack_owning_a_strategy_is_reported(catalog: StrategyCatalog) -> None:
    assert catalog.pack_of("ema_trend").name == "trend"
    assert catalog.pack_of("not_a_strategy") is None


# ------------------------------------------------------------------ behaviour


@pytest.mark.parametrize("scale", [0.0, 1.0, -1.0])
def test_every_strategy_scores_a_finite_series(catalog: StrategyCatalog, context, scale) -> None:
    """Every rule must produce a bounded conviction series once indicators warm up.

    NaNs inside the warm-up window are correct — a 200-bar EMA has no value on
    bar 3 — and the consumers drop or fill them. What must never happen is a NaN
    *after* warm-up: the ensemble fills those with zero, so a rule that broke
    would look like a rule with no opinion.
    """
    shifted = StrategyContext(bars=context.bars, features=context.features * (1.0 + scale * 0.01))
    for strategy in catalog.all():
        scores = strategy.score(shifted)
        assert len(scores) == len(shifted.features), strategy.name

        values = scores.to_numpy(dtype="float64")
        warmed = values[len(values) // 2 :]
        assert np.isfinite(warmed).all(), f"{strategy.name} produced non-finite scores after warm-up"
        finite = values[np.isfinite(values)]
        assert np.abs(finite).max() <= 1.0 + 1e-9, f"{strategy.name} exceeded unit conviction"


def test_every_strategy_handles_an_empty_frame(catalog: StrategyCatalog) -> None:
    empty_bars = generate_candles(days=1, seed=3).iloc[0:0]
    empty_context = StrategyContext(bars=empty_bars, features=pd.DataFrame(index=empty_bars.index))
    for strategy in catalog.all():
        assert strategy.score(empty_context).empty, strategy.name


def test_each_strategy_is_distinguishable(catalog: StrategyCatalog) -> None:
    """Two rules sharing a name would silently shadow each other in the catalog."""
    names = catalog.names()
    assert len(names) == len(set(names))


def test_strategies_produce_a_latest_signal_or_none(catalog: StrategyCatalog, context) -> None:
    for strategy in catalog.all():
        result = strategy._latest_signal(context, threshold=0.15)
        if result is not None:
            assert result.strategy == strategy.name
            assert -1.0 <= result.score <= 1.0
            assert result.direction.value in {"UP", "DOWN"}


def test_ensemble_blends_in_the_unit_interval(catalog: StrategyCatalog, context) -> None:
    scores = catalog.ensemble().score(context)
    values = scores.dropna().to_numpy(dtype="float64")
    assert len(values) > 0
    assert np.abs(values).max() <= 1.0
    assert np.isfinite(values).all()


# ------------------------------------------------------- capability-based ML


class StubPredictor:
    """A stand-in forecast capability, so the ML pack can be built without training.

    The curve is shaped like the one the trainer fits — parallel ``edges`` and
    ``moves`` arrays, not a flat mapping — and is steep at low conviction, which
    is what makes the cost gate testable: a modest edge is expected to precede a
    move large enough to be worth taking.
    """

    is_ready = True
    artifacts = {
        1: {
            "horizon": 1,
            "feature_names": ["ret_1"],
            "move_curve": {
                "edges": [0.05, 0.2, 0.5, 1.0],
                "moves": [8.0, 12.0, 18.0, 25.0],
                "mean_move_bps": 11.0,
            },
            "hurdle_bps": 5.0,
        }
    }

    def __init__(self, probability: float = 0.80) -> None:
        self.probability = probability

    def predict_frame(self, features: pd.DataFrame, horizon: int) -> pd.Series:
        return pd.Series(self.probability, index=features.index, dtype="float64")


def kernel_with_stub_forecast(probability: float = 0.80, hurdle_bps: float = 5.0) -> Kernel:
    kernel = Kernel(Settings())
    stub = StubPredictor(probability)
    stub.artifacts[1]["hurdle_bps"] = hurdle_bps
    kernel.register(
        PluginEntry(
            manifest=PluginManifest(name="stub", kind=PluginKind.FORECAST, provides=("forecast",)),
            build=lambda ctx, **params: stub,
            module="tests.stub_forecast",
        )
    )
    for entry in Kernel.bootstrap(Settings()).registry.find(PluginKind.STRATEGY):
        kernel.register(entry)
    return kernel


def test_ml_pack_builds_once_a_forecast_capability_exists(bars) -> None:
    """The pack links to the model through a capability, not an import."""
    kernel = kernel_with_stub_forecast()
    catalog = StrategyCatalog.from_kernel(kernel)
    assert "ml" in catalog.names()
    assert ML_PACK not in catalog.skipped


def test_ml_strategy_gates_on_the_cost_hurdle(bars) -> None:
    """A confident forecast of a move too small to pay for the round trip is silent."""
    features = build_features(bars)
    context = StrategyContext(bars=bars, features=features)

    cheap = StrategyCatalog.from_kernel(kernel_with_stub_forecast(0.60, hurdle_bps=5.0))
    expensive = StrategyCatalog.from_kernel(kernel_with_stub_forecast(0.60, hurdle_bps=50.0))

    assert cheap.get("ml").score(context).abs().sum() > 0
    assert expensive.get("ml").score(context).abs().sum() == 0


def test_ml_strategy_is_silent_within_the_noise_band(bars) -> None:
    """A probability inside min_edge means no opinion, not a tiny one."""
    features = build_features(bars)
    context = StrategyContext(bars=bars, features=features)
    catalog = StrategyCatalog.from_kernel(kernel_with_stub_forecast(0.51, hurdle_bps=0.0))
    assert catalog.get("ml").score(context).abs().sum() == 0


def test_ml_strategy_scales_with_conviction(bars) -> None:
    features = build_features(bars)
    context = StrategyContext(bars=bars, features=features)
    weak = StrategyCatalog.from_kernel(kernel_with_stub_forecast(0.60, hurdle_bps=0.0))
    strong = StrategyCatalog.from_kernel(kernel_with_stub_forecast(0.95, hurdle_bps=0.0))

    weak_score = float(weak.get("ml").score(context).abs().max())
    strong_score = float(strong.get("ml").score(context).abs().max())
    assert strong_score > weak_score > 0
