from __future__ import annotations

import pandas as pd

from core.settings import Settings
from kernel import Kernel
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
    engine._recompute(issue_research=True)

    assert engine.online_lab is not None
    assert engine.performance is not None
    assert engine.online_lab.pending_count == 9  # 3 algorithms x next 3 bars

    target = engine.history.index[-1] + pd.Timedelta(minutes=1)
    engine._observe_research(
        {"ts": target.to_pydatetime(), "close": float(engine.last_price * 1.001)}
    )

    snapshot = engine.snapshot().research["ai"]
    assert snapshot["pending"] == 6
    assert all(card["samples"] == 1 for card in snapshot["scorecards"])
    assert {card["algorithm"] for card in snapshot["scorecards"]} == {
        "online_logistic",
        "passive_aggressive",
        "gaussian_nb",
    }
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
