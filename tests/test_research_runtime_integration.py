from __future__ import annotations

import pandas as pd
import pytest

from core.settings import Settings
from kernel import Kernel
from plugins.forecasts.online_research.algorithms import ALGORITHMS
from plugins.sources.simulated.series import generate_candles
from runtime.engine import Engine


def test_engine_issues_and_scores_each_online_algorithm_after_research_is_enabled(
    tmp_path,
) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "models",
        log_dir=tmp_path / "logs",
    )
    engine = Engine(Kernel.bootstrap(settings, with_entry_points=False)).bootstrap(
        generate_candles(days=4, seed=901)
    )
    engine.enable_research()

    assert engine.online_lab is not None
    assert engine.performance is not None
    assert engine.online_lab.pending_count == len(ALGORITHMS) * 3
    records = engine.online_lab.records(limit=len(ALGORITHMS) * 3)
    assert any(
        name.startswith("strategy__")
        for record in records
        for name in record.features
    )
    assert any(
        name.startswith("strategy_combo")
        for record in records
        for name in record.features
    )

    target = engine.history.index[-1] + pd.Timedelta(minutes=1)
    engine._observe_research(
        {"ts": target.to_pydatetime(), "close": float(engine.last_price * 1.001)}
    )

    snapshot = engine.snapshot().research["ai"]
    assert snapshot["pending"] == len(ALGORITHMS) * 2
    assert all(card["samples"] == 1 for card in snapshot["scorecards"])
    assert {card["algorithm"] for card in snapshot["scorecards"]} == set(ALGORITHMS)
    assert snapshot["policy"] == {
        "buy_probability": 0.56,
        "sell_probability": 0.44,
        "min_trust_for_action": 0.15,
        "cost_bps": settings.cost_hurdle_bps(),
    }
    assert all(signal["target_at"] for signal in snapshot["signals"].values())
    engine.close()


def test_option_engine_supplies_index_anchor_to_statistical_plugins(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path, model_dir=tmp_path / "models")
    kernel = Kernel.bootstrap(settings, with_entry_points=False)
    index = Engine(kernel).bootstrap(generate_candles(days=4, seed=902))
    option = Engine(
        kernel,
        instrument_key="SIM|OPTION",
        anchor_provider=index.aligned_close,
    ).bootstrap(generate_candles(days=4, seed=903))

    statistical = {
        signal.strategy: signal for signal in option.signals if signal.strategy.startswith("anchor_")
    }
    assert statistical
    assert all(signal.meta["state"] != "N/A" for signal in statistical.values())


def test_snapshot_exposes_strategy_calculations_and_indicators(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path, model_dir=tmp_path / "models")
    engine = Engine(Kernel.bootstrap(settings, with_entry_points=False)).bootstrap(
        generate_candles(days=4, seed=904)
    )

    snapshot = engine.snapshot()
    ema = next(signal for signal in snapshot.signals if signal.strategy == "ema_trend")
    assert ema.meta["raw_score"] == pytest.approx(engine._latest_scores["ema_trend"])
    assert ema.meta["active_threshold"] == 0.15
    assert ema.meta["category"] == "trend"
    assert ema.meta["description"]
    assert {
        "ema_9",
        "ema_21",
        "ema_50",
        "ema_200",
        "supertrend",
        "vwap",
        "macd_hist",
        "rsi_14",
        "adx_14",
        "donchian_position",
    } <= set(snapshot.indicators)
