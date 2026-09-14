from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta

from typer.testing import CliRunner

import cli
from core.settings import Settings
from core.types import Direction, Signal
from plugins.advisory.performance_ledger import StrategyPerformanceLedger
from plugins.forecasts.online_research import OnlineResearchLab

NOW = datetime(2026, 9, 14, 9, 15, tzinfo=UTC)


def _record_research(settings: Settings) -> None:
    research = settings.data_dir / "research"
    research.mkdir(parents=True)
    with sqlite3.connect(research / "simulation.sqlite3") as database:
        database.execute(
            """CREATE TABLE forecasts (
            instrument TEXT, timeframe INTEGER, target TEXT, horizon INTEGER,
            predicted_close REAL, actual_close REAL, error_bps REAL, context TEXT,
            status TEXT)"""
        )
        database.execute(
            "INSERT INTO forecasts VALUES (?,?,?,?,?,?,?,?,?)",
            ("TEST", 1, NOW.isoformat(), 1, 101.0, 101.0, 0.0, "{}", "hit"),
        )

    lab = OnlineResearchLab(
        research / "online_ai" / "simulation" / "test.sqlite3",
        algorithms=("online_logistic",),
        min_trust_for_action=0.0,
    )
    lab.issue(
        {"momentum": 1.0},
        timestamp=NOW,
        anchor_price=100.0,
        horizon_min=1,
        instrument="TEST",
    )
    lab.observe(timestamp=NOW + timedelta(minutes=1), price=101.0, instrument="TEST")

    ledger = StrategyPerformanceLedger(research / "strategies.sqlite3", min_trust_samples=1)
    ledger.issue(
        [Signal(NOW, "test_strategy", Direction.UP, 0.8, "test")],
        timestamp=NOW,
        anchor_price=100.0,
        horizon_min=1,
        instrument="TEST",
        cost_bps=2.0,
    )
    ledger.observe(timestamp=NOW + timedelta(minutes=1), price=101.0, instrument="TEST")


def test_research_prints_every_persisted_scorecard_to_scrollback(tmp_path, monkeypatch) -> None:
    settings = Settings(root=tmp_path, data_dir=tmp_path / "data", model_dir=tmp_path / "models")
    _record_research(settings)
    monkeypatch.setattr(cli, "_settings", lambda: settings)

    result = CliRunner().invoke(cli.app, ["research", "--offline"])

    assert result.exit_code == 0, result.output
    assert "Forward outcomes" in result.output
    assert "Online AI · causal quality" in result.output
    assert "Latest AI research predictions" in result.output
    assert "Strategies · causal quality" in result.output
    assert "test_strategy" in result.output
    assert "hypothetical" in result.output


def test_research_json_has_machine_readable_ai_and_strategy_metrics(tmp_path, monkeypatch) -> None:
    settings = Settings(root=tmp_path, data_dir=tmp_path / "data", model_dir=tmp_path / "models")
    _record_research(settings)
    monkeypatch.setattr(cli, "_settings", lambda: settings)

    result = CliRunner().invoke(cli.app, ["research", "--offline", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["mode"] == "simulation"
    assert payload["ai_scorecards"][0]["samples"] == 1
    assert payload["strategy_scorecards"][0]["net_pnl_bps"] == 98.0
    assert payload["strategy_outcomes"] == {"scored": 1}


def test_research_rejects_unknown_section(tmp_path, monkeypatch) -> None:
    settings = Settings(root=tmp_path, data_dir=tmp_path / "data")
    monkeypatch.setattr(cli, "_settings", lambda: settings)

    result = CliRunner().invoke(cli.app, ["research", "--section", "magic"])

    assert result.exit_code == 2
    assert "section must be one of" in result.output
